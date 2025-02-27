// npm init -y
// npm install express cors axios
// this is used to create the proxy serve localy so that the giev code will work fine at we will 
// put it to port 5000 and in script.js we use this proxt 5000 port to access the leetcode website
//node server.js
const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
app.use(cors());
app.use(express.json());

app.post("/api/leetcode", async (req, res) => {
    const { username } = req.body;

    if (!username) {
        return res.status(400).json({ error: "Username is required" });
    }

    try {
        const response = await axios.post(
            "https://leetcode.com/graphql",
            {
                query: `
                    query getUserProfile($username: String!) {
                        matchedUser(username: $username) {
                            username
                            submitStats: submitStatsGlobal {
                                acSubmissionNum {
                                    difficulty
                                    count
                                }
                                totalSubmissionNum {
                                    difficulty
                                    submissions
                                }
                            }
                        }
                        allQuestionsCount {
                            difficulty
                            count
                        }
                    }
                `,
                variables: { username },
            },
            {
                headers: { "Content-Type": "application/json" },
            }
        );

        res.json(response.data);
    } catch (error) {
        console.error("Error fetching LeetCode data:", error.message);
        res.status(500).json({ error: "Failed to fetch data from LeetCode" });
    }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
