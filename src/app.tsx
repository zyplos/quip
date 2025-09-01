/** biome-ignore-all lint/style/noNonNullAssertion: shut up */
import type { QueryResponse } from "../Types";

const searchButton = document.getElementById(
  "searchButton"
)! as HTMLButtonElement;
const resultsContainer = document.getElementById("results")!;
const loadingElement = document.getElementById("loading")!;
const searchBox = document.getElementById("searchBox")! as HTMLInputElement;

async function fireSearchRequest() {
  showLoading();
  resultsContainer.innerHTML = "";

  const response = await fetch("/api/search", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      query: searchBox.value,
    }),
  });

  const data: QueryResponse = await response.json();

  for (const record of data.results) {
    const filename = record.filename;
    const similarity = record.similarity;

    const imageUrl = `/images/${filename}`;

    const resultElement = document.createElement("div");
    resultElement.classList.add("result-item");

    const imageElement = document.createElement("img");
    imageElement.src = imageUrl;
    imageElement.alt = record.filename;

    const paragraphElement = document.createElement("p");
    paragraphElement.textContent = `Similarity: ${similarity.toFixed(2)}`;

    resultElement.appendChild(imageElement);
    resultElement.appendChild(paragraphElement);

    resultsContainer.appendChild(resultElement);
  }

  hideLoading();
}

searchButton.addEventListener("click", fireSearchRequest);

searchBox.addEventListener("keypress", (event) => {
  if (event.key === "Enter") {
    event.preventDefault(); // Prevent form submission if applicable
    searchButton.click(); // Simulate a click on the button
  }
});

//

function showLoading() {
  searchButton.disabled = true;
  loadingElement.classList.add("show");
}

function hideLoading() {
  searchButton.disabled = false;
  loadingElement.classList.remove("show");
}
