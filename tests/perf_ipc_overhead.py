import timeit
import uuid
import queue
import threading

def gen_uuid_msg_id():
    return uuid.uuid4().bytes

def alloc_queue():
    return queue.Queue()

def alloc_event():
    return threading.Event()

def nop_fn():
    return 1

def implicit_lambda():
    foo = 1
    def return_foo():
        return foo
    return return_foo()

nop_fn = timeit.timeit("nop_fn()", globals=globals(), number=100000)
gen_uuid_time = timeit.timeit("gen_uuid_msg_id()", globals=globals(), number=100000)
alloc_queue_time = timeit.timeit("alloc_queue()", globals=globals(), number=100000)
alloc_event_time = timeit.timeit("alloc_event()", globals=globals(), number=100000)
implicit_lambda_time = timeit.timeit("implicit_lambda()", globals=globals(), number=100000)

print(f"nop_fn: {nop_fn:.6f} seconds per: {nop_fn/100000}")
print(f"gen_uuid_time: {gen_uuid_time:.6f} seconds per: {gen_uuid_time/100000}")
print(f"alloc_queue_time: {alloc_queue_time:.6f} seconds per: {alloc_queue_time/100000}")
print(f"alloc_event_time: {alloc_event_time:.6f} seconds per: {alloc_event_time/100000}")
print(f"implicit_lambda_time: {implicit_lambda_time:.6f} seconds per: {implicit_lambda_time/100000}")
