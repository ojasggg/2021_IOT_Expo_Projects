import os
from gtts import gTTS

myText = "Text to speech test completed"
language = "en"
output = gTTS(text=myText, lang=language, slow=False)
output.save("output.mp3")
os.system("start output.mp3")