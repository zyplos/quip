// https://huggingface.co/docs/transformers.js/api/pipelines#module_pipelines.FeatureExtractionPipeline
// https://huggingface.co/docs/transformers.js/api/pipelines#pipelinesimagefeatureextractionpipeline
// https://huggingface.co/docs/transformers.js/main/en/api/models#modelssiglipvisionmodel
//

import { pipeline } from "@huggingface/transformers";

// const image_feature_extractor = await pipeline(
//   "image-feature-extraction",
//   "Xenova/siglip-base-patch16-224",
//   { dtype: "q4", device: "dml" }
// );

// const url =
//   "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png";
// const features = await image_feature_extractor(url, {
//   pool: "mean",
//   normalize: true,
// });

// const extractor = await pipeline(
//   "image-feature-extraction",
//   "Xenova/siglip-base-patch16-224"
// );
// const result = await extractor(
//   "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png",
//   {
//     pool: true,
//   }
// );

const textFeatureExtractor = await pipeline(
  "feature-extraction",
  "Xenova/siglip-base-patch16-224",
  { dtype: "q4", device: "dml" }
);

("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cats.png");
const features = await textFeatureExtractor("a cat", {
  pooling: "mean",
  normalize: true,
});

console.log(features);
