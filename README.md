# 🎥 Video Analyzer - Football Tracking  

## 📌 Overview  
This project is a football video analysis tool that tracks players and the ball, keeping track of ball possession. The implementation is based on the following tutorial and repository:  

📺 **Tutorial:** [Football Analysis with YOLO](https://www.youtube.com/watch?v=neBZ6huolkg&t=4172s)  
📂 **Reference Repository:** [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis)  

## 🛠 Features  
- ⚽ **Player and Ball Tracking** using YOLO  
- 📊 **Ball Possession Analysis** to determine which player/team controls the ball  
- 🔍 **Experimental Kalman Filter Integration** to improve ball tracking (results showed no significant difference)  

## 🔬 Implementation Details  
- **Object Detection:** YOLO was used to detect players and the ball  
- **Tracking:** Applied a tracking algorithm to maintain player and ball identities over time  
- **Ball Possession:** Logic implemented to determine which player has control of the ball  
- **Kalman Filters:** Experimented with Kalman filtering to refine ball tracking but observed no noticeable improvement  

## 📌 Notes  
This project was implemented following the referenced tutorial and repository, with some modifications and experiments added to explore alternative tracking methods.  
