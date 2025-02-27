document.addEventListener("DOMContentLoaded", () => {
    const searchButton = document.getElementById("searchButton")
    const usernameInput = document.getElementById("userNameInput")
  
    const easyProgressCircle = document.querySelector(".easyProgress")
    const mediumProgressCircle = document.querySelector(".mediumProgress")
    const hardProgressCircle = document.querySelector(".hardProgress")
  
    const easyLabel = document.getElementById("easyLabel")
    const mediumLabel = document.getElementById("mediumLabel")
    const hardLabel = document.getElementById("hardLabel")
  
    const cardStatsContainer = document.querySelector(".cards")
  
    // Validate username
    function validateUsername(username) {
      if (username.trim() === "") {
        alert("Username should not be empty")
        return false
      }
      const regex = /^[a-zA-Z0-9_]{3,25}$/;
      const isMatching = regex.test(username)
      if (!isMatching) {
        alert("Invalid Username")
      }
      return isMatching
    }
  
    async function fetchUserDetails(username) {
        try {
            searchButton.textContent = "Searching...";
            searchButton.disabled = true;
    
            const response = await fetch("http://localhost:5000/api/leetcode", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({ username }),
            });
    
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
    
            const data = await response.json();
            if (data.error) {
                throw new Error(data.error);
            }
    
            displayUserData(data.data);
        } catch (error) {
            console.error("Error fetching user details:", error);
            alert("Failed to fetch user details. Please try again later.");
        } finally {
            searchButton.textContent = "Search";
            searchButton.disabled = false;
        }
    }
    
  
    function updateProgress(solved, total, label, circle) {
      const progressDegree = (solved / total) * 100
      circle.style.setProperty("--progress-degree", `${progressDegree}%`)
      label.textContent = `${solved}/${total}`
    }
  
    function displayUserData(data) {
      const totalQues = data.allQuestionsCount[0].count
      const totalEasyQues = data.allQuestionsCount[1].count
      const totalMediumQues = data.allQuestionsCount[2].count
      const totalHardQues = data.allQuestionsCount[3].count
  
      const solvedTotalEasyQues = data.matchedUser.submitStats.acSubmissionNum[1].count
      const solvedTotalMediumQues = data.matchedUser.submitStats.acSubmissionNum[2].count
      const solvedTotalHardQues = data.matchedUser.submitStats.acSubmissionNum[3].count
  
      updateProgress(solvedTotalEasyQues, totalEasyQues, easyLabel, easyProgressCircle)
      updateProgress(solvedTotalMediumQues, totalMediumQues, mediumLabel, mediumProgressCircle)
      updateProgress(solvedTotalHardQues, totalHardQues, hardLabel, hardProgressCircle)
  
      const cardsData = [
        { label: "Overall Submissions", value: data.matchedUser.submitStats.totalSubmissionNum[0].submissions },
        { label: "Overall Easy Submissions", value: data.matchedUser.submitStats.totalSubmissionNum[1].submissions },
        { label: "Overall Medium Submissions", value: data.matchedUser.submitStats.totalSubmissionNum[2].submissions },
        { label: "Overall Hard Submissions", value: data.matchedUser.submitStats.totalSubmissionNum[3].submissions },
      ]
  
      cardStatsContainer.innerHTML = ""
      cardsData.forEach((card) => {
        const cardElement = document.createElement("div")
        cardElement.className = "card"
        cardElement.innerHTML = `<strong>${card.label}</strong><p>${card.value}</p>`
        cardStatsContainer.appendChild(cardElement)
      })
    }
  
    searchButton.addEventListener("click", () => {
      const username = usernameInput.value
      if (validateUsername(username)) {
        fetchUserDetails(username)
      }
    })
  })
  
  