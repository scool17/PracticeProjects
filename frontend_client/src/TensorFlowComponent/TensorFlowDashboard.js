import { useState, useEffect } from "react";


export default function TensorFlowDashboard() {

    const [dashboardProps, setDashboardProps] = useState({
      healthifyHeatmap: "",
      healthifyLoss: "",
      healthifyAccuracy: "",
      healthifySequentialModelSummary: "",
      irisFunctionalModelSummary: "",
    });

    useEffect(() => {
      setDashboardProps(prevProps => ({
        ...prevProps,
        healthifyHeatmap: "/tensorflow/healthify_heatmap",
        healthifyLoss: "/tensorflow/healthify_loss",
        healthifyAccuracy: "/tensorflow/healthify_accuracy",
      }));

      fetch("/tensorflow/healthify_sequential_model_summary")
      .then((res) => res.text())
      .then((data) => setDashboardProps(prevProps => ({...prevProps, healthifySequentialModelSummary: data})))
      .catch((err) => console.error("Error loading summary:", err));

      fetch("/tensorflow/iris_functional_model_summary")
      .then((res) => res.text())
      .then((data) => setDashboardProps(prevProps => ({...prevProps, irisFunctionalModelSummary: data})))
      .catch((err) => console.error("Error loading summary:", err));
  }, []);

    return (
        <div>
            <h2 > Welcome to the TensorFlow Dashboard</h2>
            <div style={{ display: "flex", justifyContent: "space-between" }}>
            <div>
                <h4 style={{margin: 0}}> Correlation Heatmap </h4>
                {dashboardProps.healthifyHeatmap ? (
        <img key={dashboardProps.healthifyHeatmap}
        src={`${dashboardProps.healthifyHeatmap}?${Date.now()}`} alt="Correlation Heatmap" style={{ width: "100%", maxWidth: "600px" }} />
      ) : (
        <p>Loading...</p>
      )}
            </div>
            <div>
                {/* <h4 style={{margin: 0}}> Correlation Heatmap </h4> */}
                {dashboardProps.healthifyLoss ? (
        <img key={dashboardProps.healthifyLoss} src={`${dashboardProps.healthifyLoss}?${Date.now()}`} alt="Loss" style={{ width: "100%", maxWidth: "600px" }} />
      ) : (
        <p>Loading...</p>
      )}
            </div>
            <div>
                {/* <h4 style={{margin: 0}}> Correlation Heatmap </h4> */}
                {dashboardProps.healthifyAccuracy ? (
        <img key={dashboardProps.healthifyAccuracy} src={`${dashboardProps.healthifyAccuracy}?${Date.now()}`} alt="Accuracy" style={{ width: "100%", maxWidth: "600px" }} />
      ) : (
        <p>Loading...</p>
      )}
            </div>  
            <div
      className="html-content"
      dangerouslySetInnerHTML={{ __html: dashboardProps.healthifySequentialModelSummary }}
    />
    </div>
            <div
      className="html-content"
      dangerouslySetInnerHTML={{ __html: dashboardProps.irisFunctionalModelSummary }}
    />
        </div>
    )
}