import {
  AutoModel,
  AutoProcessor,
  RawImage,
  matmul,
} from "@huggingface/transformers";

// Load processor and model
const model_id = "jinaai/jina-clip-v2";
const processor = await AutoProcessor.from_pretrained(model_id);
const model = await AutoModel.from_pretrained(model_id, {
  dtype: "fp16" /* e.g., "fp16", "q8", or "q4" */,
});

// Prepare inputs
const urls = [
  "https://i.ibb.co/nQNGqL0/beach1.jpg",
  "https://i.ibb.co/r5w8hG8/beach2.jpg",
];
const images = await Promise.all(urls.map((url) => RawImage.read(url)));
const sentences = [
  "غروب جميل على الشاطئ", // Arabic
  "海滩上美丽的日落", // Chinese
  "Un beau coucher de soleil sur la plage", // French
  "Ein wunderschöner Sonnenuntergang am Strand", // German
  "Ένα όμορφο ηλιοβασίλεμα πάνω από την παραλία", // Greek
  "समुद्र तट पर एक खूबसूरत सूर्यास्त", // Hindi
  "Un bellissimo tramonto sulla spiaggia", // Italian
  "浜辺に沈む美しい夕日", // Japanese
  "해변 위로 아름다운 일몰", // Korean
];

// Encode text and images
const inputs = await processor(sentences, images, {
  padding: true,
  truncation: true,
});
const { l2norm_text_embeddings, l2norm_image_embeddings } = await model(inputs);

// Encode query (text-only)
const query_prefix = "Represent the query for retrieving evidence documents: ";
const query_inputs = await processor(
  query_prefix + "beautiful sunset over the beach"
);
const { l2norm_text_embeddings: query_embeddings } = await model(query_inputs);

// Compute text-image similarity scores
const text_to_image_scores = await matmul(
  query_embeddings,
  l2norm_image_embeddings.transpose(1, 0)
);
console.log("text-image similarity scores", text_to_image_scores.tolist()[0]); // [0.29530206322669983, 0.3183615803718567]

// Compute image-image similarity scores
const image_to_image_score = await matmul(
  l2norm_image_embeddings[0],
  l2norm_image_embeddings[1]
);
console.log("image-image similarity score", image_to_image_score.item()); // 0.9344457387924194

// Compute text-text similarity scores
const text_to_text_scores = await matmul(
  query_embeddings,
  l2norm_text_embeddings.transpose(1, 0)
);
console.log("text-text similarity scores", text_to_text_scores.tolist()[0]); // [0.5566609501838684, 0.7028406858444214, 0.582255482673645, 0.6648036241531372, 0.5462006330490112, 0.6791588068008423, 0.6192430257797241, 0.6258729100227356, 0.6453716158866882]
