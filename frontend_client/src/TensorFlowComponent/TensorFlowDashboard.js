import { useState, useEffect } from "react";


export default function TensorFlowDashboard() {

    const [imageUrl, setImageUrl] = useState("");

    useEffect(() => {
      setImageUrl("/tensorflow/correlation_heatmap"); // Flask API endpoint
    }, []);

    return (
        <div>
            <h2 > Welcome to the TensorFlow Dashboard</h2>
            <div>
                <h4 style={{margin: 0}}> Correlation Heatmap </h4>
                {imageUrl ? (
        <img src={imageUrl} alt="Correlation Heatmap" style={{ width: "100%", maxWidth: "600px" }} />
      ) : (
        <p>Loading...</p>
      )}
            </div>
        </div>
    )
}