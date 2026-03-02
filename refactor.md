## refactor notes 

> this is all in the context of migration to tinygrad. this library isn't really meant to be used as a standalone library. 

1. do we still need the atexit handler? naturally, the program should always end at device.close, if you press ctrl+C at the profiler, it will just run into device.close, right? so i think that can be axed. tinygrad provides this for us. 
2. we should remove *all* caching logic, tinygrad provides that for us as well. 
3. in compiler.py, we can pretty safely assume that whatever programs we write will compile, so maybe we can remove all the warning, error, and other flags? i want the most straightforward and trimmed compilation path possible. 
4. _firmware_skip_cores() is a dumb function, this shouldn't exist. we should always upload firmware, no skipping cores 
5. we do need to write a cache for just the firmware and the CQ kernels. those don't change and can be safely cached on first compile. use `.tt-cache` to save the kernels. you will need to save the regular firmware kernel, and the profiler firmware kernel. and then the CQ kernel. the CQ is never compiled with profiling enabled, so that only has one version. 
