import {
  AutoProcessor,
  RawImage,
  AutoModel,
  cos_sim,
  matmul,
  type Tensor,
} from "@huggingface/transformers";

import { JINA_V2_MODEL_NAME } from "@/transformers/config";

const processor = await AutoProcessor.from_pretrained(JINA_V2_MODEL_NAME);
const model = await AutoModel.from_pretrained(JINA_V2_MODEL_NAME, {
  dtype: "fp16",
});

async function generateImageEmbedding(path: string): Promise<Tensor> {
  const image = await RawImage.read(path);
  const imageInputs = await processor(null, image);
  const { l2norm_image_embeddings } = await model(imageInputs);

  return l2norm_image_embeddings;
}

async function generateTextEmbedding(text: string): Promise<Tensor> {
  const textInputs = await processor(text);
  const { l2norm_text_embeddings } = await model(textInputs);

  return l2norm_text_embeddings;
}

const imageEmb = await generateImageEmbedding(
  "C:/devspace/frontend/quip/images/test.jpg"
);

const textEmb = await generateTextEmbedding(
  "Represent the query for retrieving evidence documents: a cat"
);

console.log(imageEmb.size);
console.log(textEmb.size);

// const scored = await matmul(textEmb, imageEmb.transpose(1, 0));

console.log("COS", cos_sim(imageEmb.tolist()[0], textEmb.tolist()[0]));
// console.log("MAT", scored.item());

// MAT 0.19692830741405487
