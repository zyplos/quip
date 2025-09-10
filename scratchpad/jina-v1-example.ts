import {
  AutoTokenizer,
  CLIPTextModelWithProjection,
  AutoProcessor,
  CLIPVisionModelWithProjection,
  RawImage,
  cos_sim,
} from "@huggingface/transformers";

// Load tokenizer and text model
const tokenizer = await AutoTokenizer.from_pretrained("jinaai/jina-clip-v1");
const text_model = await CLIPTextModelWithProjection.from_pretrained(
  "jinaai/jina-clip-v1"
);

// Load processor and vision model
const processor = await AutoProcessor.from_pretrained(
  "Xenova/clip-vit-base-patch32"
);
const vision_model = await CLIPVisionModelWithProjection.from_pretrained(
  "jinaai/jina-clip-v1"
);

// Run tokenization
const texts = ["A blue cat", "A red cat"];
const text_inputs = tokenizer(texts, { padding: true, truncation: true });

// Compute text embeddings
const { text_embeds } = await text_model(text_inputs);

// Read images and run processor
const urls = [
  "https://i.pinimg.com/600x315/21/48/7e/21487e8e0970dd366dafaed6ab25d8d8.jpg",
  "https://i.pinimg.com/736x/c9/f2/3e/c9f23e212529f13f19bad5602d84b78b.jpg",
];
const image = await Promise.all(urls.map((url) => RawImage.read(url)));
const image_inputs = await processor(image);

// Compute vision embeddings
const { image_embeds } = await vision_model(image_inputs);

//  Compute similarities
console.log(cos_sim(text_embeds[0].data, text_embeds[1].data)); // text embedding similarity
console.log(cos_sim(text_embeds[0].data, image_embeds[0].data)); // text-image cross-modal similarity
console.log(cos_sim(text_embeds[0].data, image_embeds[1].data)); // text-image cross-modal similarity
console.log(cos_sim(text_embeds[1].data, image_embeds[0].data)); // text-image cross-modal similarity
console.log(cos_sim(text_embeds[1].data, image_embeds[1].data)); // text-image cross-modal similarity
