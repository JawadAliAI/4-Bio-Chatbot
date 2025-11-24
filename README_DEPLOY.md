# Deployment Instructions for Render

## Prerequisites
1. A [Render](https://render.com) account.
2. A GitHub or GitLab account.
3. Your code pushed to a repository.

## Steps to Deploy

1. **Push your code to GitHub/GitLab**
   - Ensure `Dockerfile`, `requirements.txt`, `app.py`, `chatbot.html`, and the `vosk-model-*` folders are in your repository.
   - *Note: If the Vosk models are very large, you might need to use Git LFS or download them in the Dockerfile, but the small English one and medium Arabic one should fit in a standard repo.*

2. **Create a New Web Service on Render**
   - Go to your Render Dashboard.
   - Click **New +** -> **Web Service**.
   - Connect your repository.

3. **Configure the Service**
   - **Name**: Choose a name (e.g., `dr-healbot`).
   - **Runtime**: Select **Docker**.
   - **Region**: Choose the one closest to you.
   - **Branch**: `main` (or your branch name).

4. **Environment Variables**
   - Scroll down to "Environment Variables" and add:
     - Key: `GOOGLE_API_KEY`
     - Value: `Your_Google_Gemini_API_Key`

5. **Deploy**
   - Click **Create Web Service**.
   - Render will build your Docker image (this may take a few minutes).
   - Once deployed, you will get a URL (e.g., `https://dr-healbot.onrender.com`).

## Important Notes
- **Persistence**: On Render's free tier, the filesystem is ephemeral. This means `patient_histories` and chat logs saved to disk will be **lost** when the app restarts. For permanent storage, you should use a database (like MongoDB or PostgreSQL) or a Render Disk (requires paid plan).
- **FFmpeg**: The included `Dockerfile` installs FFmpeg, so audio processing will work.
- **Models**: Ensure the Vosk model folders (`vosk-model-small-en-us-0.15` and `vosk-model-ar-mgb2-0.4`) are present in the root directory of your repository so they are copied into the Docker image.
