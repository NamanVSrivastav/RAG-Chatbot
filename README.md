# RAG-Chatbot
# RAG-Enhanced Chatbot Project
This project is an AI internship assignment to create a topic-specific chatbot augmented with advanced features such as Retrieval-Augmented Generation (RAG), vector database storage, advanced prompt engineering, and query preprocessing. The bot is designed to maintain context across turns, retrieve relevant information efficiently, and handle queries with enhanced accuracy and reliability.

# Technologies Used
# 1. Language Model (LLM)
Model: Biomistral 7B
Reason for Choice: Biomistral 7B is a lightweight, versatile language model with a solid performance-to-compute ratio, making it suitable for context maintenance and efficient response generation. With its healthcare training foundation, Biomistral provides relevant responses for health-related or similarly complex topics, enhancing the chatbot’s relevance in the chosen topic domain.
# 2. Embeddings
Embedding Model: NeuML/pubmedbert-base-embeddings
Reason for Choice: This model is pre-trained on scientific literature, making it ideal for parsing documents in specialized fields like healthcare, legal information, or technical domains. It converts text into embeddings that capture context and meaning, which are essential for effective vector retrieval in RAG systems.
# 3. Vector Database
Database: Qdrant
Reason for Choice: Qdrant is an efficient, open-source vector database that works well with large datasets and supports fast similarity searches, making it suitable for high-quality RAG performance. The system uses Qdrant for storing and retrieving relevant document chunks based on user queries, significantly enhancing the chatbot’s ability to pull contextually appropriate information from a large dataset.
# 4. Retrieval-Augmented Generation (RAG)
Overview: RAG combines retrieval with generation, allowing the chatbot to pull relevant information from a pre-defined dataset and incorporate it into responses. This enhances the model’s ability to answer specific, fact-based questions more accurately.
Implementation: User queries are embedded and matched to the most relevant text chunks stored in Qdrant, which are then provided to the LLM as context, producing accurate and contextually rich responses.
# 5. Advanced Prompt Engineering
System Prompt: The system prompt establishes the chatbot's role, behavior, and constraints, making it more responsive to the task domain.
Prompt Templates: Created multiple task-specific prompt templates (e.g., summarization, Q&A) to guide the chatbot's responses effectively across common user query types.
Few-Shot Learning: Includes examples in each prompt to reinforce the chatbot’s intended behavior, improving response consistency and quality.
# 6. Query Processing with Agentic AI
Preprocessing Agent: A preprocessing agent corrects user query spelling (using TextBlob) and categorizes queries based on intent (e.g., identifying if a query is a “summary” or “general” type). This preprocessing step ensures accurate handling of user input and improves the chatbot’s ability to interpret queries effectively.

# Summary of Approach
The chatbot progresses from a basic question-answering system to a sophisticated AI assistant with specialized retrieval and prompt engineering, handling queries with advanced AI techniques and a reliable, well-chosen tech stack. This setup allows the chatbot to deliver high-quality, accurate responses within a specialized domain while maintaining a robust user experience
