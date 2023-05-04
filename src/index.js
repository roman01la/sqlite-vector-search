const { OpenAIApi, Configuration } = require("openai");
const sqlite3 = require("better-sqlite3");
const { HierarchicalNSW } = require("hnswlib-node");
const { Tiktoken } = require("@dqbd/tiktoken/lite");
const cl100k_base = require("@dqbd/tiktoken/encoders/cl100k_base.json");

require("dotenv").config();

function encode(values) {
  const buffer = Buffer.alloc(values.length * 4);
  for (let i = 0; i < values.length; i++) {
    buffer.writeFloatLE(values[i], i * 4);
  }
  return buffer;
}

function decode(blob) {
  const values = new Array(blob.length / 4);
  for (let i = 0; i < values.length; i++) {
    values[i] = blob.readFloatLE(i * 4);
  }
  return values;
}

const configuration = new Configuration({
  apiKey: process.env.OPENAI_KEY,
});
const openai = new OpenAIApi(configuration);

const db = sqlite3("test.db");

db.exec("create table if not exists vectors (embedding blob, document text);");

async function createEmbedding(docs) {
  const r = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input: docs,
  });
  return r.data.data.map((d) => d.embedding);
}

async function insert(docs) {
  const vectors = await createEmbedding(docs);
  vectors.forEach((vector, idx) => {
    db.prepare("insert into vectors values (?, ?)").run(
      encode(vector),
      docs[idx]
    );
  });
}

function hnswlibSearch(vectors, knn, query) {
  const index = new HierarchicalNSW("l2", vectors[0].length);
  index.initIndex(vectors.length);
  vectors.forEach((v, idx) => {
    index.addPoint(v, idx);
  });
  return index.searchKnn(query, knn);
}

async function search(query, prompt, topK = 10) {
  console.log("Searching for relevant documents...");

  const [vector] = await createEmbedding([query]);

  const entries = db.prepare("select * from vectors").all();
  const embeddings = entries.map(({ embedding }) => decode(embedding));
  const documents = hnswlibSearch(embeddings, topK, vector).neighbors.map(
    (i) => entries[i].document
  );

  let content = documents.join("\n\n");
  const docs = slice(content, 3000);

  if (docs.length > 1) {
    console.log(`Summarizing ${docs.length} documents...`);
    const resp = await Promise.all(
      docs.map((doc) =>
        openai.createChatCompletion({
          model: "gpt-3.5-turbo",
          messages: [
            {
              role: "user",
              content: `Summarize the following document so it fits into 1/${docs.length} of ChatGPT's tokens limit: ${doc}`,
            },
          ],
        })
      )
    );
    content = resp
      .map((resp) => resp.data.choices[0].message.content)
      .join("\n\n");
  } else {
    content = docs[0];
  }

  console.log("Looking for an answer...");
  const completion = await openai.createChatCompletion({
    model: "gpt-3.5-turbo",
    messages: [{ role: "user", content: `${prompt}\n\n${content}` }],
  });

  return completion.data.choices[0].message.content;
}

function slice(doc, maxTokens) {
  const encoding = new Tiktoken(
    cl100k_base.bpe_ranks,
    cl100k_base.special_tokens,
    cl100k_base.pat_str
  );

  const tokens = encoding.encode(doc);
  const td = new TextDecoder();

  const result = [];
  for (let i = 0; i < tokens.length; i += maxTokens) {
    const slice = tokens.slice(i, i + maxTokens);
    result.push(td.decode(encoding.decode(slice)));
  }

  encoding.free();

  return result;
}

const query = process.argv[2];

if (query) {
  (async function main() {
    const ret = await search(
      query,
      `Act as a friendly knowledge base search system. Below you are given a question and knowledge base in markdown format.
          Some sections of the knowledge base are in a format of <question> and <answer>, use them to find an answer to a similar question.
          Your reply should only consist of an answer and a suggestion on how to resolve the issue if applicable.

        Answer the following question based in provided info: ${query}
        Knowledge base:`
    );
    console.log(ret);
  })();
} else {
  process.stdin.on("data", (data) => {
    const docs = slice(data.toString(), 2000);
    insert(docs).then(() => console.log(`Inserted ${docs.length} documents`));
  });
}
