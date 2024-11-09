async function swapFaces() {
    const sourceUrl = document.getElementById('sourceUrl').value;
    const targetUrl = document.getElementById('targetUrl').value;
    
    try {
        const response = await fetch('/swap-faces', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                source_image_url: sourceUrl,
                target_image_url: targetUrl
            })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const imageUrl = URL.createObjectURL(blob);
            document.getElementById('result').src = imageUrl;
        } else {
            const error = await response.json();
            alert(error.detail);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
} 