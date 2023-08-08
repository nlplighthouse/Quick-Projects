import os 
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import whisper 
from langchain.llms import OpenAI 
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain 
import tiktoken
from pprint import pprint 
import yt_dlp 

# A function to download a youtube video 
def download_mp4_from_youtube(url, file_name):
    # Set the options for the download
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
        'outtmpl': file_name,
        'quiet': True,
    }

    # Download the video file
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(url, download=True)
        
        

# A function to split the video into segments 
# extract the starttime and endtime of each section from video_timestamps
def segment_video(video_file_name, video_timestamps, video_ending_timestamp, segment_file_path):
    
    # Check if video_segments directory exists, if not create it (to store the video segments)
    if not os.path.exists(segment_file_path):
        os.mkdir(segment_file_path)
    
    
    def timestamp_to_seconds(timestamp):
        """Converts a timestamp in the format HH:MM:SS to seconds."""
        h, m, s = timestamp.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)


    for i in range(len(video_timestamps)-1):
        
        # calculate the startime in seconds from the timestamp 
        starttime = timestamp_to_seconds(video_timestamps[i][0])
        endtime = timestamp_to_seconds(video_timestamps[i+1][0])
        timestamp_title = video_timestamps[i][1]
        
        ffmpeg_extract_subclip(video_file_name, starttime, endtime, targetname=f'{segment_file_path}/{timestamp_title}.mp4')
        
        # check if i is the last index in the list and compute starttime and endtime 
        if i == len(video_timestamps)-2:
            startime = endtime
            endtime = timestamp_to_seconds(video_ending_timestamp)
            timestamp_title = video_timestamps[i+1][1]
            ffmpeg_extract_subclip(video_file_name, starttime, endtime, targetname=f'{segment_file_path}/{timestamp_title}.mp4')
    


# A function to transcribe the video segments

def transcribe_videos(transcriber_model, video_files_path: str, transcript_file_path: str):
    
    # check if video_trascripts folder exists, if not create it
    if not os.path.exists(transcript_file_path):
        os.makedirs(transcript_file_path)
        
    for video_segment in os.listdir(video_files_path):
        
        trascript_result = transcriber_model.transcribe(f'{video_files_path}/{video_segment}')
        text_transcript = trascript_result['text']
        
        title = video_segment.split('.')[0]
        
        # save to text file in video_transcripts folder
        with open(f'{transcript_file_path}/{title}.txt', 'w') as f:
            f.write(text_transcript)
            

# A function to summarize the video segments

def summarize(llm, transcript_file_path: str, summary_file_path: str):
    
    
    # A prompt template for summarization 
    prompt_template = """
    You are a YouTube video summarizer assistant. Your task is to summarize an INTERVIEW TRANSCRIPT.
    
    Your summaries should focus on the key takeaways based on the VIDEO_TITLE provided below.
    
    You should provide the interviewee's key messages in bullet points. (not more than 5 bullet points)
    
    The interviewee name is Max, so you should respond as if you are Max.
    
    
    VIDEO_TITLE:
    ```{video_title}```
    
    INTERVIEW_TRANSCRIPT:
    ```{interview_transcript}```
    
    Summary:    
    """
    
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['video_title', 'interview_transcript']
    )
    
    summarizer_chain = LLMChain(
        llm = llm,
        prompt=prompt 
    )
    
    
    
    
    # check if video_summaries folder exists, if not create it
    if not os.path.exists(summary_file_path):
        os.makedirs(summary_file_path)
    
    for video_segment in os.listdir(transcript_file_path):
        
        file_path = os.path.join(transcript_file_path, video_segment)
        
        # read the text file
        with open(file_path, 'r') as f:
            text = f.read()
            
        title = video_segment.split('.')[0]
            
        # summarize the text
        summary = summarizer_chain.run(
            {
                'video_title': title,
                'interview_transcript': text
            }
        )

        
        # save to text file in video_summaries folder
        with open(f'{summary_file_path}/{title}.txt', 'w') as f:
            f.write(summary)
            


if __name__ == '__main__':
    
    # STEP 0: Set up the variables
    
    # A link to the video
    video_url = 'https://www.youtube.com/watch?v=3951WpStCQg'
    
    # sections of the video taken from YT description
    video_timestamps = [
        ('00:00:00', 'Introduction'),
        ('00:09:19', 'How he got a remote job through Twitter'),
        ('00:14:06', 'Introduction to Ray'),
        ('00:18:52', 'Reinforcement learning'),
        ('00:23:56', 'Key lessons on integrating customer feedback'),
        ('00:35:12', 'Flaws in data science job titles'),
        ('00:45:51', 'How to be irreplaceable as a data scientist'),
        ('00:48:55', 'An unconventional career path as a data scientist'),
        ('01:12:24', 'Productivity and work-life balance'),
        ('01:28:10', 'Advice for building a personal brand'),
    ]
    
    # timestamp for the end of the video
    video_ending_timestamp = '1:53:28'
    
    # Initialize whisper model
    whisper_model = whisper.load_model('base')
    
    # A file name to save the video as
    video_file_name = 'daliana-max-interview.mp4'
    
    # A folder to store the video segments
    video_segments_folder = './video_segments'
    
    # A folder to store the video transcripts
    transcripts_folder = './video_transcripts'
    
    # A folder to store the video summaries
    summaries_folder = './video_summaries'
    
    # Initialize LLM and prompt template
    llm = OpenAI(model_name='gpt-4', temperature=0)
    
    
    # Step - 1: Download the video
    download_mp4_from_youtube(video_url, video_file_name)
    
    
    # STEP - 2: Segment the video
    segment_video(video_file_name, 
                  video_timestamps, 
                  video_ending_timestamp, 
                  video_segments_folder)
    
    # STEP - 3: Transcribe the video segments
    transcribe_videos(whisper_model,
                      video_segments_folder, 
                      transcripts_folder)
    
    # Step - 4: Summarize the video segments
    summarize(llm, transcripts_folder, summaries_folder)
    
    
    
    
    
    
    
    
    
