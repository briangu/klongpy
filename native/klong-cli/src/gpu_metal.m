#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdio.h>
#include <string.h>

@interface KGMetal : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) NSArray<id<MTLComputePipelineState>> *pipelines;
@end

@implementation KGMetal
@end

static void message(char *dst, size_t cap, NSString *msg) {
    snprintf(dst, cap, "%s", msg.UTF8String ?: "unknown Metal error");
}

static bool complete(id<MTLCommandBuffer> command, char *error, size_t cap) {
    [command commit];
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusCompleted) {
        return true;
    }
    message(error, cap, command.error.localizedDescription ?: @"Metal command did not complete");
    return false;
}

void *kg_metal_open(const char *source, size_t len, int64_t ordinal, char *error, size_t cap) {
    @autoreleasepool {
        id<MTLDevice> device;
        if (ordinal < 0) {
            device = MTLCreateSystemDefaultDevice();
        } else {
            NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();
            device = (uint64_t)ordinal < devices.count ? devices[(NSUInteger)ordinal] : nil;
        }
        if (!device) {
            message(error, cap, @"selected Metal device is unavailable");
            return NULL;
        }
        MTLCompileOptions *options = [MTLCompileOptions new];
#if __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
        if ([options respondsToSelector:@selector(setMathMode:)]) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunguarded-availability-new"
            options.mathMode = MTLMathModeSafe;
#pragma clang diagnostic pop
        } else
#endif
        {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }
        NSString *text = [[NSString alloc] initWithBytes:source length:len encoding:NSUTF8StringEncoding];
        NSError *failure = nil;
        id<MTLLibrary> lib = [device newLibraryWithSource:text options:options error:&failure];
        if (!lib) {
            message(error, cap, failure.localizedDescription ?: @"could not compile Metal kernels");
            return NULL;
        }
        NSArray<NSString *> *names = @[@"elementwise", @"sum_blocks", @"sum_columns", @"gradient_tile", @"gradient_collapse"];
        NSMutableArray<id<MTLComputePipelineState>> *pipelines = [NSMutableArray arrayWithCapacity:names.count];
        for (NSString *name in names) {
            id<MTLFunction> function = [lib newFunctionWithName:name];
            if (!function) {
                message(error, cap, [@"missing Metal kernel " stringByAppendingString:name]);
                return NULL;
            }
            id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&failure];
            if (!pipeline) {
                message(error, cap, failure.localizedDescription ?: @"could not create Metal pipeline");
                return NULL;
            }
            [pipelines addObject:pipeline];
        }
        KGMetal *metal = [KGMetal new];
        metal.device = device;
        metal.queue = [device newCommandQueue];
        metal.pipelines = pipelines;
        if (!metal.queue) {
            message(error, cap, @"could not create Metal command queue");
            return NULL;
        }
        return (__bridge_retained void *)metal;
    }
}

void kg_metal_release(void *object) {
    id retained = (__bridge_transfer id)object;
    (void)retained;
}

void *kg_metal_upload(void *context, const void *data, size_t bytes, char *error, size_t cap) {
    @autoreleasepool {
        KGMetal *metal = (__bridge KGMetal *)context;
        id<MTLBuffer> staging = [metal.device newBufferWithBytes:data length:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> output = [metal.device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        if (!staging || !output) {
            message(error, cap, @"could not allocate Metal upload buffers");
            return NULL;
        }
        id<MTLCommandBuffer> command = [metal.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (!blit) {
            message(error, cap, @"could not start Metal upload");
            return NULL;
        }
        [blit copyFromBuffer:staging sourceOffset:0 toBuffer:output destinationOffset:0 size:bytes];
        [blit endEncoding];
        return complete(command, error, cap) ? (__bridge_retained void *)output : NULL;
    }
}

bool kg_metal_download(void *context, void *buffer, void *data, size_t bytes, char *error, size_t cap) {
    @autoreleasepool {
        KGMetal *metal = (__bridge KGMetal *)context;
        id<MTLBuffer> input = (__bridge id<MTLBuffer>)buffer;
        id<MTLBuffer> staging = [metal.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!staging) {
            message(error, cap, @"could not allocate Metal readback buffer");
            return false;
        }
        id<MTLCommandBuffer> command = [metal.queue commandBuffer];
        id<MTLBlitCommandEncoder> blit = [command blitCommandEncoder];
        if (!blit) {
            message(error, cap, @"could not start Metal readback");
            return false;
        }
        [blit copyFromBuffer:input sourceOffset:0 toBuffer:staging destinationOffset:0 size:bytes];
        [blit endEncoding];
        if (!complete(command, error, cap)) {
            return false;
        }
        memcpy(data, staging.contents, bytes);
        return true;
    }
}

void *kg_metal_dispatch(void *context, uint32_t kernel, void *a, void *b, size_t bytes,
                        uint32_t groups, uint32_t width, const uint32_t *params, size_t param_bytes,
                        char *error, size_t cap) {
    @autoreleasepool {
        KGMetal *metal = (__bridge KGMetal *)context;
        id<MTLComputePipelineState> pipeline = metal.pipelines[kernel];
        id<MTLBuffer> output = [metal.device newBufferWithLength:bytes options:MTLResourceStorageModePrivate];
        if (!output) {
            message(error, cap, @"could not allocate Metal compute buffer");
            return NULL;
        }
        if (width > pipeline.maxTotalThreadsPerThreadgroup) {
            message(error, cap, @"Metal kernel exceeds this device's threadgroup limit");
            return NULL;
        }
        id<MTLCommandBuffer> command = [metal.queue commandBuffer];
        id<MTLComputeCommandEncoder> dispatch = [command computeCommandEncoder];
        if (!dispatch) {
            message(error, cap, @"could not start Metal compute pass");
            return NULL;
        }
        [dispatch setComputePipelineState:pipeline];
        [dispatch setBuffer:(__bridge id<MTLBuffer>)a offset:0 atIndex:0];
        [dispatch setBuffer:(__bridge id<MTLBuffer>)(b ?: a) offset:0 atIndex:1];
        [dispatch setBuffer:output offset:0 atIndex:2];
        [dispatch setBytes:params length:param_bytes atIndex:3];
        [dispatch dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [dispatch endEncoding];
        return complete(command, error, cap) ? (__bridge_retained void *)output : NULL;
    }
}
