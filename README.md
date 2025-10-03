Got it ✅ Here’s a clean, beginner-friendly **README.md** for your GitHub repo. It explains your SEO Consultant app, how to install, run, and contribute, and also highlights your Flow + CrewAI design.

---

# 📊 SEO Consultant

An AI-powered **Search Engine Optimization (SEO) Consultant** built with **[CrewAI](https://docs.crewai.com/)** and **Streamlit**, developed by *Vatsal Kumar Singh*.

This app helps you:

* Analyze your website for SEO performance
* Identify top competitors in your niche
* Research competitor strategies and keywords
* Generate a comprehensive, professional SEO report

---

## 🚀 Features

* **Website Analysis** – evaluates site structure, weaknesses, and areas for SEO improvement.
* **Competitor Discovery** – finds 5–10 relevant competitors with URLs.
* **Competitor Research** – provides comparative insights & keyword suggestions.
* **SEO Report Writing** – delivers a 2–3 page structured report.
* **Interactive UI** – built with Streamlit for easy usage.
* **Flow-based Orchestration** – powered by CrewAI’s Flow system for modular and extensible pipelines.

---

## 🛠️ Tech Stack

* [Python 3.10+](https://www.python.org/)
* [CrewAI](https://docs.crewai.com/) – Agent orchestration & Flows
* [Streamlit](https://streamlit.io/) – UI layer
* [OpenAI](https://platform.openai.com/) – LLMs for analysis & writing
* [Serper](https://serper.dev/) – Google search API
* [Pydantic](https://docs.pydantic.dev/) – State management

---

## 📂 Project Structure

```
SEO-consultant/
│── agents.py        # Defines AI agents (website analyst, competitor analyst, etc.)
│── task.py          # Defines tasks assigned to each agent
│── flows.py         # Orchestrates workflow using CrewAI Flows
│── main.py          # Streamlit app entry point
│── requirements.txt # Dependencies
│── README.md        # Project documentation
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/SEO-consultant.git
cd SEO-consultant
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set API Keys**
   You need:

* OpenAI API Key → [Get here](https://platform.openai.com/)
* Serper API Key → [Get here](https://serper.dev/)

Either export them in your terminal:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export SERPER_API_KEY="your_serper_api_key"
```

Or enter them directly in the Streamlit sidebar when running the app.

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run main.py
```

* Enter your **website URL**
* Provide **OpenAI + Serper API keys**
* Choose a model (e.g. `gpt-4o`, `gpt-4.1-mini`)
* Get your full SEO analysis & competitor insights!

---

## 🧠 How It Works

1. **Flow Orchestration (`flows.py`)**

   * Starts with website analysis
   * Finds competitors
   * Researches competitors
   * Writes final report

2. **Agents (`agents.py`)**

   * Each agent is specialized (analyst, researcher, writer).

3. **Tasks (`task.py`)**

   * Define the work each agent must do.

4. **Streamlit (`main.py`)**

   * Provides the user interface.

---

## 📌 Example Output

* Overall SEO score
* Site weaknesses & improvement actions
* Competitor list with URLs
* Comparative analysis of competitors
* Suggested keywords to rank higher
* Final professional SEO report

---

## 👤 Author

**Vatsal Kumar Singh**

---
