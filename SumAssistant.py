import time

import openai
import argparse
import os

import you_get
from you_get.extractors import (
    imgur,
    magisto,
    youtube,
    missevan,
    acfun,
    bilibili,
    soundcloud,
    tiktok,
    twitter,
    miaopai
)


parser = argparse.ArgumentParser(description='Sum Assistant.')
parser.add_argument('--link', type=str, required=True)
parser.add_argument('--temp_dir', type=str, default='./sum_assistant_temp')
parser.add_argument("--remove_temp", type=bool, default=True)
parser.add_argument("--ffmpeg_path", type=str, default="/Users/qhu/Downloads/ffmpeg")

args = parser.parse_args()

temp_video_dir = args.temp_dir + "/video"
temp_audio_dir = args.temp_dir + "/audio"
temp_transcription = args.temp_dir + "/transcription"

openai.api_key = os.environ.get("OPENAI_API_KEY")

# create temp dir
if not os.path.exists(args.temp_dir):
    os.makedirs(args.temp_dir)
if not os.path.exists(temp_video_dir):
    os.makedirs(temp_video_dir)
if not os.path.exists(temp_audio_dir):
    os.makedirs(temp_audio_dir)
if not os.path.exists(temp_transcription):
    os.makedirs(temp_transcription)

start_time = time.time()

# download video
# os.system("you-get -O {} {}".format(temp_video_dir, args.link))
you_get.extractors.bilibili.download(args.link, output_dir=temp_video_dir, merge=False) # caption / danmaku


# traverse temp video folder, convert video to audio
for root, dirs, files in os.walk(temp_video_dir):
    for file in files:
        if file == ".DS_Store":
            continue
        audio_file_name = file.replace(".", "_") + ".mp3"
        cmd = "{} -i \"{}\" -y \"{}\"".format(
            args.ffmpeg_path,
            os.path.join(temp_video_dir, file).replace("\"", "\\\""),
            os.path.join(temp_audio_dir, audio_file_name).replace("\"", "\\\"")
        )

        print(cmd)
        os.system(cmd)

transcriptions = []

# traverse temp audio folder, convert audio to transcription
for root, dirs, files in os.walk(temp_audio_dir):
    for file in files:
        audio_file_path = os.path.join(temp_audio_dir, file)
        audio_file = open(audio_file_path, "rb")
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
        transcriptions.append(transcription)
        transcription_file_path = os.path.join(temp_transcription, file.replace(".", "_") + ".txt")
        with open(transcription_file_path, "w") as f:
            f.write(transcription.text)
        audio_file.close()

content_part = ""
# merge transcriptions
for ind, transcription in enumerate(transcriptions):
    current_text = transcription.text
    content_part += "\n[{}]".format(ind + 1) + "\n" + current_text + "\n"

# generate summary

system_msg_part = (
                    "现在需要进行视频提取出的字幕文本总结以及要点提取任务。请你尽力提供准确和简洁的总结，以及对于其中的要点进行列举。"
                    "要求如下：1.如果内容较长，请先分析文本的层次，然后对于每一个部分进行总结和提取要点，并且依照层次分别叙述；"
                    "2.对于需要进行处理的内容，"
                    "(1)如果是说明性质，则着重提取其说明的对象描述、步骤或流程、内涵和外延；"
                    "(2)如果是议论性质着重于其提出的论点、论据论证；"
                    "(3)如果是故事性的，请简要描述故事的起因经过结果以及作者想要表达的内容；"
                    "(4)如果是抒情性的，请简要描述作者的情感、情绪、感受以及作者想要表达的内容；(5)其他类型的内容请自行分析。"
                    )


response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo-16k",
  messages=[
    {"role": "system", "content": system_msg_part},
    {"role": "user", "content": "下面是文本的内容:" + content_part}
  ]
)

stop_time = time.time()
print("time cost: {}".format(stop_time - start_time))

print(response['choices'][0]['message']['content'])


if args.remove_temp:
    os.system("rm -rf {}".format(args.temp_video_dir))
    os.system("rm -rf {}".format(args.temp_audio_dir))
