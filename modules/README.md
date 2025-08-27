## Install whisper.cpp
from [whisper.cpp](https://github.com/ggml-org/whisper.cpp)

Then, download one of the Whisper models converted in ggml format. For example
```
sh ./models/download-ggml-model.sh base.en
```
Now build the whisper-cli example and transcribe an audio file like this:

```
# Now, configure the build again using cmake. The key is to add the flag -DBUILD_SHARED_LIBS=OFF. This tells CMake not to build the .so shared library and instead link everything statically.
cmake -B build -DBUILD_SHARED_LIBS=OFF

make -C build

# transcribe an audio file
./build/bin/whisper-cli -f samples/jfk.wav
```