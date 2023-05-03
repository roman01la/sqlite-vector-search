# sqlite-vector-search
A demo of KNN scanning in SQLite

- Uses OpenAI embeddings
- Splits documents into chunks based on specified amount of tokens
- Stores vectors in SQLite as buffers
- Scans vectors using cosine similarity
- Finds related documents and answers questions via ChatGPT API

## Setup
1. Install NPM deps
2. Put your OpenAI key into `.env` file: `OPENAI_KEY={key}`
3. Pipe text into `node src/index.js` to embed and store it in SQLite
4. Run `node src/index.js "your question"` to get an answer based on data in the db
