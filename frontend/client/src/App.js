import './App.css';
import React, { useState, useEffect } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList } from "recharts";


function App() {

  const [data, setData] = useState([]);
  const [loss, setLoss] = useState([]);

  const labelColors = {
    0: "#FF5733", // Red
    1: "#33FF57", // Green
    2: "#3357FF", // Blue
};

const groupedData = data.reduce((acc, point) => {
  if (!acc[point.label]) {
      acc[point.label] = [];
  }
  acc[point.label].push(point);
  return acc;
}, {});

  useEffect(() => {
    fetch("/predict").then(
      res => res.json()).then(
        data => {setLoss(data)
          console.log("Data is", data)
  })
        }, [])

  useEffect(() => {
    fetch("/show").then((response) => response.json())
        .then((json) => {
            // Convert {x1, x2, y} to {x, y}
            const formattedData = json.map(item => ({
                x: item.x1,  
                y: item.x2,
                label: item.y
            }));
            setData(formattedData);
        })
        .catch((error) => console.error("Error fetching data:", error));
}, []);

  return (
    <>
    <div style={{ width: "50vh", height: "50vh", margin: "auto" }}>
    <ResponsiveContainer width="100%" height="100%" aspect={1}>
            <ScatterChart>
                <CartesianGrid />
                <XAxis type="number" dataKey="x" name="X Value" />
                <YAxis type="number" dataKey="y" name="Y Value" />
                <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                {/* <Scatter name="Data Points" data={data} fill="#8884d8" >
                  <LabelList dataKey="label" position="top" />
                </Scatter> */}
                {Object.entries(groupedData).map(([label, points]) => (
                    <Scatter key={label} name={label} data={points} fill={labelColors[label] || "#8884d8"} />
                ))}
            </ScatterChart>
        </ResponsiveContainer>
        </div>
    </>
  );
}

export default App;
