# Text Similarity and Embedding Technique App

A simple app that enables you to play around with the three most common text similarity measures along with a few embedding methods.

![gif of the app](demo.gif "App Demo")

You can find an in-depth explaination of the different similarity measures and text embedding methods [here](https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python).

To run the web app locally, try:
```
docker build . -t similarity_app
docker run -p 8501:8501 similarity_app
```