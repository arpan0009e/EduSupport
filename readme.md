# 🎓 EduSupport AI
### Fine-Tuned AI Assistant for Engineering Students

EduSupport AI is an AI-powered student assistant built using **Qwen2.5-1.5B-Instruct**, fine-tuned with **QLoRA and LoRA**. It provides simple, concise, and branch-specific explanations for engineering students.

## 📸 Demo Preview

<!-- Upload your application screenshots here -->

![EduSupport AI Home](screenshots/home.png)

![EduSupport AI Chat](screenshots/chat.png)

## ✨ Features

- Supports five engineering branches.
- Fine-tuned Qwen2.5-1.5B-Instruct model.
- Parameter-efficient fine-tuning using QLoRA and LoRA.
- Simple and concise technical explanations.
- Structured responses using Python.
- Interactive Streamlit chatbot.
- Local GPU inference with 4-bit quantization.

## 🎓 Supported Branches

- AI & Machine Learning
- Computer Science & Engineering
- Electronics & Communication Engineering
- Mechanical Engineering
- Civil Engineering

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Language | Python |
| Base Model | Qwen2.5-1.5B-Instruct |
| Fine-Tuning | LoRA, QLoRA, PEFT |
| Frameworks | PyTorch, Transformers, TRL |
| Quantization | BitsAndBytes |
| Frontend | Streamlit |

##  Model Training

- Dataset: 500 synthetic engineering Q&A records.
- Training samples: 400.
- Validation samples: 100.
- Fine-tuning: QLoRA with 4-bit NF4 quantization.
- LoRA rank: 8.
- Epochs: 3.

##  Architecture

```text
Student
   |
   v
Streamlit Interface
   |
   v
Qwen2.5-1.5B-Instruct
   +
LoRA Adapter
   |
   v
Response Generation
   |
   v
Python Formatter
   |
   v
Structured Answer
```

## 📁 Project Structure

```text
EduSupport_AI/
├── app.py
├── model_loader.py
├── formatter.py
├── requirements.txt
├── README.md
│
├── model/
│   └── EduSupport_AI_Adapter/
│
└── screenshots/
```

## ⚙️ Run Locally

**1. Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/EduSupport_AI.git
cd EduSupport_AI
```

**2. Create and activate virtual environment**

```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Place the fine-tuned adapter inside the `model/` directory.**

**5. Run the application**

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.


##  Author

**Arpan Mondal**  
B.Tech — Artificial Intelligence & Machine Learning

[GitHub](https://github.com/YOUR_USERNAME) | [LinkedIn](https://www.linkedin.com/in/YOUR_LINKEDIN_USERNAME/)

---

*Developed as an AI/ML engineering project for personalized student learning.*