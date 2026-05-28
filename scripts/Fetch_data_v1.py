import requests
import pandas as pd
import os
from  dotenv import load_dotenv 

load_dotenv()

API_KEY=os.getenv("YOUTUBE_API_KEY")

def get_comments(video_id, tool_name):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    
    comments = []
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": 100,
        "key": API_KEY
    }

    while True:
        response = requests.get(url, params=params)
        data = response.json()

        for item in data["items"]:
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            
            comment = snippet["textDisplay"]
            likes = snippet["likeCount"]

            comments.append({
                "comment_text": comment,
                "likes": likes,
                "video_id": video_id,
                "tool_name": tool_name
            })

        if "nextPageToken" in data:
            params["pageToken"] = data["nextPageToken"]
        else:
            break

    return comments


videos = [
    {"id": "VIDEO_ID_1", "tool": "ChatGPT"},
    {"id": "VIDEO_ID_2", "tool": "ChatGPT"},
    {"id": "VIDEO_ID_3", "tool": "Gemini"},
    {"id": "VIDEO_ID_4", "tool": "Gemini"},
]

all_comments = []

for video in videos:
    print(f"Fetching comments for {video['id']}")
    data = get_comments(video["id"], video["tool"])
    all_comments.extend(data)

# Convert to DataFrame
df = pd.DataFrame(all_comments)

# Save to CSV
df.to_csv("youtube_ai_comments.csv", index=False)

print("Dataset saved successfully!")