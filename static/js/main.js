// static/js/main.js
document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const predictionResultDiv = document.getElementById('predictionResult');
    const usdPriceSpan = document.getElementById('usdPrice');
    const rwfPriceSpan = document.getElementById('rwfPrice');
    const inputDataList = document.getElementById('inputDataList');
    const errorMessageDiv = document.getElementById('errorMessage');
// static/js/main.js
console.log('Main JavaScript loaded');
    // The theme toggle is now handled directly in base.html script block,
    // but leaving this for local clarity if needed for other elements.
    // const themeToggle = document.getElementById('themeToggle');
    // if (themeToggle) { // Check if element exists before adding listener
    //     themeToggle.addEventListener('click', () => {
    //         document.body.classList.toggle('dark');
    //         document.body.classList.toggle('light');
    //         if (document.body.classList.contains('light')) {
    //             document.body.classList.remove('bg-darkBase');
    //             document.body.classList.add('bg-lightBase');
    //         } else {
    //             document.body.classList.remove('bg-lightBase');
    //             document.body.classList.add('bg-darkBase');
    //         }
    //     });
    // }

    if (form) { // Only add event listener if the form exists (i.e., on index.html)
        form.addEventListener('submit', async (event) => {
            event.preventDefault(); // Prevent default form submission

            // Clear previous results and errors
            predictionResultDiv.classList.add('hidden');
            inputDataList.innerHTML = ''; // Clear previous input data list items
            errorMessageDiv.classList.add('hidden');
            errorMessageDiv.textContent = '';

            const formData = new FormData(form);
            const data = {};
            formData.forEach((value, key) => {
                if (['year', 'month', 'day_of_week', 'day_of_year'].includes(key)) {
                    data[key] = parseInt(value, 10);
                } else {
                    data[key] = value;
                }
            });

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data),
                });

                if (response.redirected) {
                    window.location.href = response.url; // Handle redirects (e.g., to login page)
                    return;
                }

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                }

                const result = await response.json();

                usdPriceSpan.textContent = `$${result.predicted_usdprice.toFixed(2)}`;
                rwfPriceSpan.textContent = `RWF ${result.predicted_rwfprice.toFixed(2)}`;

                const displayOrder = [
                    'year', 'month', 'day_of_week', 'day_of_year',
                    'admin1', 'admin2', 'market', 'category', 'commodity', 'unit', 'pricetype', 'currency'
                ];

                displayOrder.forEach(key => {
                    if (result.input_data.hasOwnProperty(key)) {
                        const listItem = document.createElement('li');
                        listItem.className = 'flex justify-between items-center';

                        const formattedKey = key
                            .replace(/_/g, ' ')
                            .replace(/\b\w/g, char => char.toUpperCase());

                        listItem.innerHTML = `<span class="font-medium text-gray-400">${formattedKey}:</span> <span class="text-white font-semibold">${result.input_data[key]}</span>`;
                        inputDataList.appendChild(listItem);
                    }
                });

                predictionResultDiv.classList.remove('hidden');

            } catch (error) {
                errorMessageDiv.textContent = `Error: ${error.message}`;
                errorMessageDiv.classList.remove('hidden');
                console.error('Prediction failed:', error);
            }
        });
    }
});