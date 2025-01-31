// Function to fetch predictions from the backend
async function getPrediction() {
    // Fetch user input
    const data = {
        Cs: parseFloat(document.getElementById('Cs').value),
        FA: parseFloat(document.getElementById('FA').value),
        MA: parseFloat(document.getElementById('MA').value),
        Cl: parseFloat(document.getElementById('Cl').value),
        Br: parseFloat(document.getElementById('Br').value),
        I: parseFloat(document.getElementById('I').value),
        model_name: document.getElementById('model-selection').value
    };

    // Validation check for empty fields or invalid inputs
    if (Object.values(data).some(value => isNaN(value))) {
        document.getElementById('result').innerText = "Please fill in all fields with valid numbers.";
        return;
    }

    // Clear previous results and show loading message
    document.getElementById('result').innerText = "Calculating prediction...";

    try {
        // Send POST request to the backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        // Check if the response is OK (status 200)
        if (!response.ok) {
            throw new Error('Failed to fetch prediction');
        }

        // Parse the JSON response from the backend
        const result = await response.json();

        // Display prediction result
        if (result.prediction) {
            document.getElementById('result').innerHTML = `
                <strong>Prediction:</strong> ${result.prediction.toFixed(4)} eV
            `;
        } else if (result.error) {
            document.getElementById('result').innerText = `Error: ${result.error}`;
        }
    } catch (error) {
        console.error("Error fetching prediction:", error);
        document.getElementById('result').innerText = "An error occurred while fetching the prediction. Please try again later.";
    }
}

// Event listener for the form submission button
document.getElementById('prediction-form').addEventListener('submit', (event) => {
    event.preventDefault();  // Prevent default form submission
    getPrediction();
});
