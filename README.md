# 🧠 Text Summarization & Sentiment Analysis API with FastAPI and MongoDB

This project provides an API endpoint to process English texts by:
- Detecting the language
- Performing sentiment analysis
- Extracting keywords
- Summarizing the text using a pre-trained transformer model
- Storing the results in a **MongoDB Atlas** database

All functionalities are exposed via a FastAPI POST endpoint.

---

## 🚀 Features

- 🌐 Language detection using `langdetect`
- 😊 Sentiment analysis via `TextBlob`
- 🧠 Keyword extraction with `spaCy` and `collections.Counter`
- ✂️ Text summarization using `facebook/bart-large-cnn` transformer via HuggingFace
- 💾 Data persistence using **MongoDB Atlas** (cloud MongoDB database)

---

## 📦 Requirements

Install all dependencies with pip:

```bash
pip install fastapi uvicorn transformers spacy textblob langdetect pymongo nltk
python -m textblob.download_corpora
python -m spacy download en_core_web_sm
````

---

## 📡 How to Run

```bash
uvicorn main:app --reload
```

Once the server is running, send a POST request to:

```
POST http://127.0.0.1:8000/summary
```

### 📝 Request Body

```json
{
  "text": "Your English text here"
}
```

---

## 🧾 Sample Response

```json
{
  "inserted_id": "64c...e5a",
  "summary_data": {
    "Sentiment": "Positive",
    "Keywords": ["AI", "OpenAI", "model", "applications", "ChatGPT"],
    "Summary": "OpenAI develops AI for humanity's benefit. Models like ChatGPT are widely used."
  }
}
```

---

## ☁️ MongoDB Atlas Integration

### What is MongoDB Atlas?

MongoDB Atlas is a fully managed cloud database developed by the creators of MongoDB. It allows you to host and scale MongoDB databases in the cloud (AWS, GCP, or Azure) and connect securely from your apps.

### 🔧 Connection URI Format:

```python
uri = "mongodb+srv://<username>:<password>@cluster0.mongodb.net/<dbname>?retryWrites=true&w=majority"
```

Replace:

* `<username>` and `<password>` with your MongoDB credentials
* `<dbname>` with your database name (e.g., `mongo`)

Ensure your cluster is configured to **accept connections from all IP addresses** or your specific IP (Colab users: use `0.0.0.0/0` for testing).

### Example Usage:

```python
client = MongoClient(uri, server_api=ServerApi(version='1'))
db = client["mongo"]  # Database
collection = db["sum_"]  # Collection
```

Each request will save the input text and its:

* Sentiment
* Extracted keywords
* Summary

---

## 📂 Project Structure

```
main.py           # FastAPI app with summarization logic
requirements.txt  # (optional) list of all dependencies
```

---

## 🔒 Note on Language Support

This project only processes English texts. Requests in other languages will return an HTTP 400 error.

---

## 📬 Future Improvements

* Add support for multiple languages
* JWT-based authentication
* Query endpoint to retrieve summaries from MongoDB

---

## 🧑‍💻 Author

Created by \[Your Name]
Feel free to contribute or report issues!

---

## 🛡 License

MIT License

```
