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
            <div>
                <h4 style={{margin: 0}}> Correlation Heatmap </h4>
                {dashboardProps.healthifyHeatmap ? (
        <img src={dashboardProps.healthifyHeatmap} alt="Correlation Heatmap" style={{ width: "100%", maxWidth: "600px" }} />
      ) : (
        <p>Loading...</p>
      )}
            </div>
            <div>
                {/* <h4 style={{margin: 0}}> Correlation Heatmap </h4> */}
                {dashboardProps.healthifyLoss ? (
        <img src={dashboardProps.healthifyLoss} alt="Loss" style={{ width: "100%", maxWidth: "600px" }} />
      ) : (
        <p>Loading...</p>
      )}
            </div>
            <div>
                {/* <h4 style={{margin: 0}}> Correlation Heatmap </h4> */}
                {dashboardProps.healthifyAccuracy ? (
        <img src={dashboardProps.healthifyAccuracy} alt="Accuracy" style={{ width: "100%", maxWidth: "600px" }} />
      ) : (
        <p>Loading...</p>
      )}
            </div>
            <div
      className="html-content"
      dangerouslySetInnerHTML={{ __html: dashboardProps.healthifySequentialModelSummary }}
    />
            <div
      className="html-content"
      dangerouslySetInnerHTML={{ __html: dashboardProps.irisFunctionalModelSummary }}
    />
        </div>
    )
}