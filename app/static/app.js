async function detectToxicity() {
    // Show the loading message
    document.getElementById("loading").style.display = "block";

    // Get the comments from the textarea
    const comments = document.getElementById("comments").value;
    const data = {
        comments: comments.split("\n")
    };

    try {
        // Send request to Flask API
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        // Parse the result
        const result = await response.json();
        document.getElementById("loading").style.display = "none"; // Hide the loading message

        // Display results
        const resultsDiv = document.getElementById("results");
        resultsDiv.innerHTML = ""; // Clear previous results

        result.predictions.forEach(prediction => {
            const resultItem = document.createElement("div");
            resultItem.classList.add("result-item");
            resultItem.innerHTML = `
                <strong>Comment:</strong> ${prediction.comment} <br>
                <strong>Toxic:</strong> ${prediction.toxic} <br>
                <strong>Severe Toxic:</strong> ${prediction.severe_toxic} <br>
                <strong>Obscene:</strong> ${prediction.obscene} <br>
                <strong>Threat:</strong> ${prediction.threat} <br>
                <strong>Insult:</strong> ${prediction.insult} <br>
                <strong>Identity Hate:</strong> ${prediction.identity_hate} <br>
            `;
            resultsDiv.appendChild(resultItem);
        });

    } catch (error) {
        document.getElementById("loading").style.display = "none";
        console.error("Error detecting toxicity:", error);
        alert("Error detecting toxicity. Please try again.");
    }
}
