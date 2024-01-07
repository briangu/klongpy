There are a few examples here for how to interoperate with Python's multiprocessing capabilities.

pool.kg - Shows how to call a function that, behind the scenes, uses multiprocessing to perform its task.  The Klong code just receives the answer after the work is done across multiple processes.

callback.kg - Shows how to marshal Klong functions to processes so that it runs "over there" and executes in a fresh Klong interpreter context.  Currently, the Klong interpreter doesn't serialize in multiprocessing calls so you need to instantiate a fresh context and run the function in new context.  This can be useful in simple map/reduce type situations, where the context of the main process isn't needed in sub-processes.

worker - Shows how to overcome the worker process context limits by loading a new Klong program in a worker process.  This allows workers to be arbitrarily complex.  Worker processes load the new Klong program and run it, returning the result as expected.


Note, any state that needs to be shared from the primary process can be serialized to the worker process in the starmap call.  

Future work: make the marshaling of state from the primary process to worker processes more seamless so that the Klong functions can be expressed in one place - the main code.  The tricky bit is mainly knowing which context to marshal to worker processes - currently this is done manually.

