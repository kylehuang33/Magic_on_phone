## Install whisper.cpp
from [whisper.cpp](https://github.com/ggml-org/whisper.cpp)

Then, download one of the Whisper models converted in ggml format. For example
```
sh ./models/download-ggml-model.sh base.en
```
Now build the whisper-cli example and transcribe an audio file like this:

```
# build the project

cmake -B build
cmake --build build -j --config Release

# transcribe an audio file
./build/bin/whisper-cli -f samples/jfk.wav
```