
import React, { useState, useEffect } from "react";

import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";


export default function NeuralNetworkDashboard() {
    
    const [data, setData] = useState([]);
    
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

    return(
        <div>
            <h2> Welcome to the TensorFlow Dashboard</h2>
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
    </div>
    )
}