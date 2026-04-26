# 🤖 Agent ReAct avec RAG et Outils - Guide Complet

---

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Concepts Fondamentaux](#concepts-fondamentaux)
3. [Architecture](#architecture)
4. [Implémentation](#implémentation)
5. [Exemples de Code](#exemples-de-code)
6. [Cas d'Usage](#cas-dusage)
7. [Bonnes Pratiques](#bonnes-pratiques)
8. [Dépannage](#dépannage)

---

## Introduction

Un **agent ReAct** est un système d'intelligence artificielle qui combine le **Raisonnement** et l'**Action** pour résoudre des problèmes complexes de manière itérative. Contrairement aux modèles de langage traditionnels qui génèrent une réponse en une seule passe, les agents ReAct peuvent:

- Analyser un problème
- Décider quelle action prendre
- Exécuter l'action
- Observer le résultat
- Répéter jusqu'à obtenir la réponse finale

---

## Concepts Fondamentaux

### 1. Le Cycle ReAct

Le cycle ReAct se compose de trois étapes principales:

**Reason (Raisonnement)**: L'agent analyse le problème et décide quelle action prendre.

**Act (Action)**: L'agent exécute l'action sélectionnée (appel d'outil, recherche, etc.).

**Observe (Observation)**: L'agent observe le résultat et l'intègre dans son contexte.

Ce cycle se répète jusqu'à ce que l'agent ait suffisamment d'informations pour répondre.

### 2. RAG (Retrieval-Augmented Generation)

Le RAG est une technique qui augmente les capacités du LLM en lui fournissant des documents pertinents récupérés d'une base de connaissances.

**Avantages du RAG:**
- Réponses plus précises et contextualisées
- Accès à des informations à jour
- Réduction des hallucinations (réponses inventées)
- Possibilité d'utiliser des documents propriétaires

### 3. Les Outils

Les outils permettent à l'agent d'interagir avec le monde extérieur. Exemples courants:

| Outil | Description | Exemple |
|-------|-------------|---------|
| Calculatrice | Effectuer des calculs | 234 * 567 |
| Recherche Web | Récupérer des infos actuelles | Météo, actualités |
| Base de Données | Accéder à des données structurées | Requêtes SQL |
| RAG | Rechercher dans des documents | Manuels, articles |
| API Externe | Intégrer des services externes | Traduction, conversion |

---

## Architecture

### Flux d'Exécution

```
┌─────────────────────────────┐
│   Entrée Utilisateur        │
└────────────┬────────────────┘
             │
             ↓
┌─────────────────────────────┐
│   LLM (Raisonnement)        │
│   "Quelle action faire?"    │
└────────────┬────────────────┘
             │
             ↓
┌─────────────────────────────┐
│   Sélection d'Outil         │
│   (Calculatrice/RAG/etc)    │
└────────────┬────────────────┘
             │
             ↓
┌─────────────────────────────┐
│   Exécution de l'Outil      │
│   Récupération des données  │
└────────────┬────────────────┘
             │
             ↓
┌─────────────────────────────┐
│   Observation du Résultat   │
│   Intégration au contexte   │
└────────────┬────────────────┘
             │
             ↓
        Réponse finale?
        /              \
      OUI              NON
      │                │
      ↓                ↓
  Retourner      Nouvelle itération
  Réponse        (Reason)
```

### Composants Clés

**LLM (Language Model)**: Le cerveau de l'agent qui prend les décisions.

**Tools**: Les actions que l'agent peut exécuter.

**Memory**: Le contexte et l'historique des actions précédentes.

**Executor**: Le composant qui exécute les outils.

---

## Implémentation

### Installation des Dépendances

```bash
pip install langchain langgraph langsmith openai faiss-cpu
```

### Configuration de Base

```python
import os
from langchain.llms import ChatOpenAI
from langchain.agents import Tool, initialize_agent

# Configurer la clé API
os.environ["OPENAI_API_KEY"] = "votre-clé-api"

# Initialiser le LLM
llm = ChatOpenAI(model="gpt-4", temperature=0)
```

---

## Exemples de Code

### Exemple 1: Agent Simple avec Outils

```python
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.llms import ChatOpenAI

# Définir une fonction pour la calculatrice
def calculator(expression: str) -> str:
    try:
        result = eval(expression)
        return f"Résultat: {result}"
    except Exception as e:
        return f"Erreur: {e}"

# Définir une fonction pour la recherche
def search_web(query: str) -> str:
    # Simulation d'une recherche web
    return f"Résultats pour '{query}': Article 1, Article 2, Article 3"

# Créer les outils
tools = [
    Tool(
        name="Calculatrice",
        func=calculator,
        description="Utile pour effectuer des calculs mathématiques"
    ),
    Tool(
        name="Recherche Web",
        func=search_web,
        description="Utile pour rechercher des informations sur le web"
    )
]

# Initialiser l'agent
agent = initialize_agent(
    tools,
    llm=ChatOpenAI(model="gpt-4"),
    agent=AgentType.REACT_DOCSTORE,
    verbose=True,
    max_iterations=5
)

# Utiliser l'agent
response = agent.run("Quel est le résultat de 234 * 567?")
print(response)
```

### Exemple 2: Intégration du RAG

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.agents import Tool

# Charger les documents
loader = TextLoader("documents.txt")
documents = loader.load()

# Diviser en chunks
splitter = CharacterTextSplitter(chunk_size=1000)
docs = splitter.split_documents(documents)

# Créer la base de vecteurs
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Créer l'outil RAG
def rag_search(query: str) -> str:
    results = vectorstore.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in results])

rag_tool = Tool(
    name="Recherche Documents",
    func=rag_search,
    description="Recherche dans la base de documents pour trouver des informations pertinentes"
)

# Ajouter à l'agent
tools.append(rag_tool)
```

### Exemple 3: Agent avec LangGraph

```python
from langgraph.graph import StateGraph, END
from langchain.agents import Tool
from typing import TypedDict, List

# Définir l'état
class AgentState(TypedDict):
    input: str
    history: List[str]
    current_action: str
    result: str
    done: bool

# Créer les nœuds
def reason_node(state: AgentState) -> AgentState:
    # Raisonnement: décider quelle action prendre
    state["current_action"] = "Analyser le problème..."
    return state

def act_node(state: AgentState) -> AgentState:
    # Action: exécuter l'outil sélectionné
    state["result"] = "Résultat de l'outil..."
    return state

def observe_node(state: AgentState) -> AgentState:
    # Observation: intégrer le résultat
    state["history"].append(state["result"])
    state["done"] = True  # Ou False pour continuer
    return state

# Créer le graphe
graph = StateGraph(AgentState)
graph.add_node("reason", reason_node)
graph.add_node("act", act_node)
graph.add_node("observe", observe_node)

# Ajouter les arêtes
graph.add_edge("reason", "act")
graph.add_edge("act", "observe")
graph.add_conditional_edges(
    "observe",
    lambda x: END if x["done"] else "reason"
)

# Compiler et exécuter
compiled_graph = graph.compile()
```

### Exemple 4: Monitoring avec LangSmith

```python
from langsmith import Client
from langchain.callbacks.langchain_callback import LangChainTracer

# Initialiser LangSmith
client = Client()

# Tracer les exécutions
tracer = LangChainTracer(project_name="mon-agent")

# Exécuter avec traçage
with client.trace_session("agent-session"):
    response = agent.run(
        "Votre question",
        callbacks=[tracer]
    )

# Consulter les traces sur https://smith.langchain.com
```

---

## Cas d'Usage

### 1. Assistant de Recherche Intelligent

Un agent qui combine recherche web + RAG pour répondre à des questions complexes.

```python
agent = initialize_agent(
    tools=[search_tool, rag_tool],
    llm=llm,
    agent=AgentType.REACT_DOCSTORE,
    verbose=True
)

response = agent.run("Expliquez les dernières avancées en IA générative")
```

### 2. Analyseur de Données

Un agent qui peut interroger des bases de données et effectuer des analyses.

```python
tools = [
    Tool(name="SQL Query", func=execute_sql),
    Tool(name="Calculatrice", func=calculator),
    Tool(name="Visualisation", func=create_chart)
]

agent = initialize_agent(tools, llm)
response = agent.run("Quel est le chiffre d'affaires moyen par région?")
```

### 3. Assistant Technique

Un agent qui utilise RAG sur une documentation technique pour aider les utilisateurs.

```python
# Charger la documentation
docs = load_documentation("api_docs/")
vectorstore = create_vectorstore(docs)

agent = initialize_agent(
    tools=[rag_tool, code_executor_tool],
    llm=llm
)

response = agent.run("Comment faire une requête GET avec authentification?")
```

### 4. Planificateur de Tâches

Un agent qui peut planifier et exécuter des séquences de tâches.

```python
tools = [
    Tool(name="Créer Tâche", func=create_task),
    Tool(name="Mettre à jour Tâche", func=update_task),
    Tool(name="Envoyer Email", func=send_email),
    Tool(name="Créer Rappel", func=create_reminder)
]

agent = initialize_agent(tools, llm)
response = agent.run("Crée une tâche pour demain, envoie un email à l'équipe et rappelle-moi")
```

---

## Bonnes Pratiques

### 1. Définir les Outils Clairement

Chaque outil doit avoir une description claire et précise pour que l'agent comprenne quand l'utiliser.

```python
Tool(
    name="Calculatrice",
    func=calculator,
    description="Effectue des calculs mathématiques. Utilisez cet outil pour les opérations arithmétiques complexes."
)
```

### 2. Limiter les Itérations

Évitez les boucles infinies en définissant un nombre maximum d'itérations.

```python
agent = initialize_agent(
    tools,
    llm,
    max_iterations=5,  # Limiter à 5 itérations
    early_stopping_method="force"
)
```

### 3. Gérer les Erreurs

Implémentez une gestion d'erreur robuste dans vos outils.

```python
def safe_tool(input: str) -> str:
    try:
        result = execute_tool(input)
        return result
    except Exception as e:
        return f"Erreur: {str(e)}"
```

### 4. Monitorer les Performances

Utilisez LangSmith pour tracer et optimiser les performances.

```python
# Consulter les métriques
# - Temps d'exécution
# - Nombre d'itérations
# - Taux de réussite
# - Coûts des appels API
```

### 5. Tester les Cas Limites

Testez votre agent avec des entrées inattendues.

```python
test_cases = [
    "Question simple",
    "Question complexe avec plusieurs étapes",
    "Question impossible à répondre",
    "Entrée vide",
    "Entrée très longue"
]

for test in test_cases:
    response = agent.run(test)
    print(f"Test: {test} → {response}")
```

---

## Dépannage

### Problème: L'agent boucle infiniment

**Solution**: Augmentez `max_iterations` ou utilisez `early_stopping_method="force"`.

### Problème: L'agent ne sélectionne pas le bon outil

**Solution**: Améliorez les descriptions des outils et testez avec des exemples.

### Problème: Réponses incohérentes

**Solution**: Réduisez la température du LLM (`temperature=0`) pour plus de cohérence.

### Problème: Coûts API élevés

**Solution**: Utilisez le caching et limitez le nombre d'itérations.

### Problème: RAG retourne des résultats non pertinents

**Solution**: Améliorez le chunking et les embeddings, ou utilisez un meilleur modèle d'embedding.

---

## Ressources Supplémentaires

- **Documentation LangChain**: https://python.langchain.com/
- **Documentation LangGraph**: https://langchain-ai.github.io/langgraph/
- **LangSmith**: https://smith.langchain.com/
- **OpenAI API**: https://platform.openai.com/docs/
- **Exemples GitHub**: https://github.com/langchain-ai/langchain

---

## Conclusion

Les agents ReAct avec RAG et outils offrent une approche puissante pour construire des systèmes d'IA intelligents et autonomes. En combinant le raisonnement, l'action et l'observation, ces agents peuvent résoudre des problèmes complexes et fournir des réponses précises et contextualisées.

**Prochaines étapes:**
1. Construisez votre premier agent
2. Expérimentez avec différents outils
3. Optimisez les performances
4. Déployez en production

Bon apprentissage! 🚀
