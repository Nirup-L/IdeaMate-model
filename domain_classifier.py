from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline

# Initialize transformer model and tokenizer for domain classification
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

# Create a classifier using transformers pipeline
nlp = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

# Domain-related keywords
domain_keywords =   {
        "IoT": ["internet of thing","internet of things", "aiot", "raspberry pi", "breadboard", "iot", "smart devices", "wearables", "sensor networks", "smart cities", "machine-to-machine", "connected devices", "home automation", "smart agriculture", "smart health", "industrial IoT", "IoT platforms", "edge computing", "wireless communication", "RFID", "BLE", "5G IoT", "IoT sensors", "IoT devices", "IoT architecture", "cloud IoT", "IoT security", "IoT applications", "smart homes", "smart grids", "IoT in healthcare", "IoT in manufacturing", "IoT in transportation", "IoT in agriculture", "IoT security", "smart farming", "connected vehicles", "IoT in logistics", "IoT analytics" , "smart" , "devices" , "circuit" , "chips" , "sender" , "receiver" , "encoder" , "decoder"],
        
        "AI": ["machine learning", "deep learning", "AI", "NLP", "neural networks", "reinforcement learning", "chatbot", "computer vision", "predictive analytics", "data science", "big data", "AI ethics", "AI applications", "generative AI", "speech recognition", "image processing", "speech synthesis", "AI models", "natural language generation", "AI algorithms", "cognitive computing", "knowledge graphs", "AI in healthcare", "AI in education", "AI in business", "AI for customer service", "AI in finance", "AI in agriculture", "AI in cybersecurity", "AI in marketing", "AI for robotics", "chatbots", "automated decision making", "AI for social good", "AI research", "intelligent systems", "AI for healthcare", "AI-driven insights", "data mining", "pattern recognition", "data preprocessing", "feature extraction", "supervised learning", "unsupervised learning", "AI platforms", "AI in e-commerce", "reinforcement learning applications", "AI tools", "model training", "AI-powered analytics", "AI technology"],
        
        "ML": ["machine learning", "algorithms", "data mining", "predictive models", "supervised learning", "unsupervised learning", "reinforcement learning", "feature selection", "deep learning", "AI", "model training", "support vector machines", "decision trees", "regression models", "clustering", "k-means", "neural networks", "random forests", "data science", "big data", "data preprocessing", "model validation", "cross-validation", "hyperparameter tuning", "deep neural networks", "model evaluation", "time series analysis", "text classification", "image classification", "semantic analysis", "neural language models", "ensemble methods", "boosting", "bagging", "autoML", "reinforcement learning", "unsupervised learning", "model deployment", "AI in business", "machine learning applications", "recommendation systems", "AI-powered prediction", "real-time analytics"],
        
        "Agriculture": ["agricultur","farming", "crop", "agriculture", "horticulture", "seeds", "irrigation", "organic farming", "agritech", "precision farming", "agriculture technology", "drones", "farm management", "pest control", "genetic modification", "agriculture robotics", "sustainable farming", "crop rotation", "soil health", "agriculture sensors", "fertilizer", "agriculture machinery", "hydroponics", "aeroponics", "vertical farming", "precision irrigation", "agriculture drones", "smart irrigation", "agriculture data", "farm automation", "bioengineering", "plant disease detection", "agriculture robotics", "livestock management", "agriculture analytics", "agriculture software", "fertilizer management", "disease management", "greenhouses", "indoor farming", "farm-to-table", "agriculture supply chain", "agriculture production", "crop yield", "bio pesticides", "agriculture market", "food security", "food systems", "smart farms", "weather prediction", "agriculture apps", "plant growth", "genetically modified organisms", "digital agriculture", "climate change in agriculture", "crop protection", "livestock breeding", "farm education", "agriculture consulting", "sustainable agriculture", "food waste", "food production", "farmers market", "agriculture infrastructure", "data-driven farming", "autonomous farming", "smart farming", "water conservation", "green technology", "agriculture sustainability", "natural farming", "soil management", "crop disease", "agriculture policy", "agriculture finance", "food quality", "bio fertilizers", "rural development", "agriculture supply chain", "agriculture marketing", "agriculture innovation", "food traceability", "harvest prediction", "agriculture exports", "agriculture processing", "drone technology", "precision farming", "rural economy", "agriculture marketing", "agriculture labor", "agriculture education", "food safety", "smart agriculture systems", "pest resistance", "agriculture research", "digital farms", "climate-smart agriculture", "agriculture startups", "agribusiness", "farm data", "agriculture management software", "crop monitoring", "farm insurance", "fertilizer use", "agriculture sensors", "agriculture systems", "agriculture productivity", "agriculture education", "sustainable farming practices", "farm-to-fork", "agriculture trade", "farm machinery", "soil sensors", "weather data", "remote sensing", "agriculture financing", "water management", "pesticide reduction", "harvest management", "agriculture infrastructure", "food demand", "plant breeding", "gmo crops", "agriculture growth", "supply chain optimization", "regenerative agriculture", "land use", "climate change adaptation", "sustainable practices", "food systems planning", "agriculture-based policies", "greenhouses", "indoor farming", "vertical farming", "organic agriculture", "crop growth", "agriculture yield"],
        
        "Clean and Green Technology": ["renewable energy", "solar energy", "wind energy", "green technology", "electric vehicles", "clean energy", "sustainability", "carbon footprint", "energy efficiency", "green buildings", "eco-friendly", "biofuels", "hydropower", "clean tech", "recycling", "sustainable manufacturing", "waste management", "sustainable transportation", "clean water technology", "climate change mitigation", "green engineering", "carbon capture", "eco-innovation", "alternative energy", "clean tech startups", "sustainable agriculture", "clean coal", "green economy", "smart grids", "energy storage", "zero emissions", "sustainable development", "green chemistry", "bioenergy", "clean transportation", "green manufacturing", "low carbon technologies", "clean industrial technologies", "green logistics", "electric cars", "energy-efficient lighting", "smart cities", "carbon-neutral", "green design", "green infrastructure", "sustainable urban planning", "waste-to-energy", "solar panels", "wind turbines", "bio-based products", "green hydrogen", "environmental conservation", "eco-friendly products", "climate tech", "green innovations", "smart energy", "sustainable mobility", "water purification", "sustainable building materials", "clean air technology", "clean cooking", "clean mobility", "energy transition", "clean tech financing", "bio-based technologies", "environmentally friendly", "sustainable practices", "energy audits", "electric grids", "energy harvesting", "resource efficiency", "green startups", "green supply chain", "eco-tourism", "environmental protection", "clean energy financing", "climate action", "green tech for agriculture", "electric buses", "clean energy systems", "clean technologies", "green logistics", "clean manufacturing", "recycled materials", "sustainable resources", "eco-design", "environmentally conscious"],
        
        "Environment": ["climate change", "conservation", "sustainability", "green energy", "ecosystem", "wildlife", "pollution", "environmental protection", "carbon footprint", "renewable resources", "environmental technology", "environmental sciences", "climate action", "green innovation", "water conservation", "environmental policy", "nature conservation", "carbon emissions", "global warming", "climate science", "air quality", "recycling", "biodiversity", "land degradation", "sustainable agriculture", "deforestation", "eco-friendly", "plastic waste", "greenhouse gases", "waste management", "environmental conservation", "sustainable development", "nature reserves", "eco-tourism", "environmental awareness", "clean energy", "green tech", "resource conservation", "ecological footprint", "ocean preservation", "green architecture", "wildlife protection", "ecosystem restoration", "air pollution", "renewable energy", "green building", "pollution control", "nature-based solutions", "green economy", "urban greening", "zero waste", "sustainable cities", "reforestation", "environmental advocacy", "pollution monitoring", "bioengineering", "landfills", "environmental sustainability", "environmental education", "biodiversity protection", "climate adaptation", "global sustainability", "pollution reduction", "conservation science", "eco-innovation", "global environmental issues", "clean water", "pollution prevention", "carbon neutrality", "green spaces", "sustainable materials", "environmental monitoring", "renewable resources", "green investments", "climate resilience", "eco-friendly products", "climate mitigation", "carbon credits", "environmental impact assessment", "carbon trading", "green certification", "ecosystem services", "eco-friendly construction", "clean air", "sustainable transport", "eco-consumerism", "recycled products", "environmental regulation", "earth sciences", "urban sustainability", "sustainable farming", "green lifestyle", "green energy solutions", "ecological restoration", "climate change action", "climate change solutions", "green jobs", "environmental economics", "global climate agreement", "climate research", "natural resource management", "carbon capture", "environmental resilience", "ecological restoration", "sustainability leadership", "eco-friendly solutions", "waste reduction", "climate policy", "conservation biology", "carbon trading", "ecological footprint", "waste management technology", "environmental engineering", "climate change mitigation", "nature conservation", "sustainable energy", "clean technologies" , "pollution" , "pollute" , "pollut"],
        
        "Blockchain": [
    "blockchain", "decentralized", "cryptocurrency", "smart contracts", "Bitcoin", "Ethereum", "NFTs", "distributed ledger", "tokenization", "DeFi", "blockchain technology", "cryptography", "peer-to-peer", "mining", "digital wallet", "public key", "private key", "blockchain platforms", "DApps", "blockchain security", "ledger", "blockchain nodes", "blockchain consensus", "scalability", "blockchain networks", "proof of work", "proof of stake", "tokenomics", "blockchain governance", "DAO", "blockchain applications", "ICO", "blockchain interoperability", "blockchain in finance", "blockchain in supply chain", "blockchain scalability", "blockchain standards", "blockchain analytics", "blockchain for healthcare", "blockchain in energy", "blockchain in logistics", "permissioned blockchain", "permissionless blockchain", "sidechains", "layer 2 solutions"
],
"Cybersecurity": [
    "cybersecurity", "ethical hacking", "penetration testing", "malware", "ransomware", "phishing", "firewall", "VPN", "data encryption", "endpoint security", "threat detection", "cybersecurity training", "zero trust", "cybersecurity tools", "intrusion detection", "intrusion prevention", "vulnerability scanning", "network security", "application security", "DDoS", "cyber attack", "social engineering", "cryptographic protocols", "cyber resilience", "digital forensics", "cybersecurity frameworks", "cybersecurity policies", "security patches", "identity management", "cloud security", "IoT security", "endpoint detection", "data breaches", "SIEM", "cybersecurity compliance", "GDPR", "SOC", "CISO", "cybersecurity awareness", "cybersecurity automation"
],
"Robotics": [
    "robotics", "autonomous robots", "humanoid robots", "industrial robots", "robot arms", "swarm robotics", "robot programming", "sensors in robotics", "robotics engineering", "robot control", "robot kinematics", "robotic process automation", "AI in robotics", "robot vision", "collaborative robots", "robotic systems", "robotics simulation", "mechatronics", "robot actuators", "mobile robots", "robot design", "surgical robots", "service robots", "robot hardware", "robotic mobility", "robot operating systems (ROS)", "autonomous navigation", "robot localization", "robot mapping", "drones", "robotics competitions", "robotics applications", "robotics in healthcare", "robotics in agriculture", "soft robotics", "teleoperation", "robot ethics"
],
"Augmented Reality & Virtual Reality": [
    "augmented reality", "virtual reality", "mixed reality", "XR", "AR applications", "VR applications", "AR/VR in gaming", "AR in retail", "VR in education", "AR/VR hardware", "AR glasses", "VR headsets", "AR/VR development", "Unity AR/VR", "Unreal Engine AR/VR", "AR markers", "VR environments", "immersive experiences", "AR overlays", "VR simulations", "haptics", "AR/VR design", "AR SDKs", "AR APIs", "VR experiences", "AR/VR for training", "AR for marketing", "spatial computing", "AR effects", "VR for healthcare", "AR/VR in real estate", "AR/VR in tourism", "AR/VR storytelling"
],
"Data Science": [
    "data analysis", "data wrangling", "data visualization", "machine learning", "predictive analytics", "statistical modeling", "data mining", "data preprocessing", "big data", "data ethics", "Python for data science", "R programming", "SQL for data science", "data engineering", "ETL", "data pipelines", "business intelligence", "descriptive statistics", "inferential statistics", "data exploration", "anomaly detection", "clustering algorithms", "data storytelling", "Tableau", "Power BI", "data dashboards", "open data", "data quality", "data governance", "time series analysis", "data lakes", "data warehouses", "data formats (CSV, JSON, XML)", "data interoperability"
],
"Quantum Computing": [
    "quantum computing", "qubits", "quantum mechanics", "quantum algorithms", "quantum entanglement", "quantum gates", "quantum circuits", "quantum supremacy", "quantum cryptography", "quantum hardware", "superposition", "quantum coherence", "quantum decoherence", "quantum teleportation", "quantum annealing", "quantum key distribution", "quantum programming", "quantum simulators", "quantum networks", "IBM Q", "Google Quantum", "quantum error correction", "quantum machine learning", "quantum optimization", "quantum chemistry", "quantum cloud computing", "quantum computing frameworks", "quantum noise", "quantum speedup"
],
"Space Technology": [
    "space exploration", "satellites", "rockets", "propulsion systems", "space missions", "space robotics", "CubeSats", "space communication", "space debris", "ISS (International Space Station)", "Mars exploration", "moon missions", "space telescopes", "space science", "space innovation", "space tourism", "asteroid mining", "space weather", "space economy", "reusable rockets", "space launch systems", "satellite imaging", "remote sensing", "space technology in agriculture", "space policy", "nanosatellites", "satellite networks", "space observatories", "deep space exploration", "orbital mechanics"
],
"Healthcare Technology": [
    "telemedicine", "electronic health records (EHR)", "wearable devices", "health monitoring", "AI in healthcare", "medical imaging", "diagnostic tools", "healthcare analytics", "health apps", "fitness trackers", "patient management systems", "healthcare IoT", "remote patient monitoring", "health informatics", "medical robots", "healthcare platforms", "digital health", "healthcare chatbots", "health data privacy", "smart hospitals", "biosensors", "medical AI", "personalized medicine", "mobile health (mHealth)", "health cybersecurity", "telehealth platforms", "healthcare APIs", "health data integration"
],
"Autonomous Vehicles": [
    "self-driving cars", "autonomous driving", "vehicle-to-vehicle (V2V) communication", "vehicle-to-infrastructure (V2I)", "ADAS (Advanced Driver Assistance Systems)", "autonomous car sensors", "LiDAR", "radar", "autonomous navigation", "GPS systems", "EVs with autonomy", "self-driving software", "autonomous truck technology", "autonomous fleet management", "self-parking vehicles", "levels of autonomy", "autonomous drones", "autonomous vehicle testing", "smart traffic management", "AV in logistics", "AV safety", "robotaxis", "self-driving AI"
],
"Artificial Intelligence": [
    "AI", "machine learning", "deep learning", "neural networks", "AI applications", "computer vision", "natural language processing", "AI ethics", "AI algorithms", "AI in healthcare", "AI in education", "AI in finance", "AI in gaming", "AI tools", "chatbots", "speech recognition", "recommendation systems", "AI models", "reinforcement learning", "AI research", "AI training data", "AI optimization", "AI in robotics", "AI for automation", "AI for business", "AI for personalization"
],
"Cloud Computing": [
    "cloud computing", "IaaS", "PaaS", "SaaS", "cloud storage", "cloud security", "AWS", "Microsoft Azure", "Google Cloud", "cloud migration", "hybrid cloud", "private cloud", "public cloud", "cloud deployment", "cloud scalability", "serverless computing", "virtual machines", "containerization", "Kubernetes", "cloud databases", "cloud architecture", "cloud cost optimization", "cloud analytics", "cloud disaster recovery", "cloud automation", "cloud compliance"
],
"E-Commerce": [
    "e-commerce", "online shopping", "e-commerce platforms", "payment gateways", "e-commerce SEO", "drop shipping", "product listing", "e-commerce analytics", "cart abandonment", "e-commerce marketing", "e-commerce apps", "e-commerce websites", "e-commerce logistics", "customer retention", "e-commerce trends", "e-commerce UI/UX", "mobile commerce", "cross-border e-commerce", "e-commerce personalization", "e-commerce automation", "affiliate marketing", "e-commerce inventory management"
],
"Education Technology (EdTech)": [
    "EdTech", "online learning", "virtual classrooms", "LMS (Learning Management Systems)", "MOOCs", "education apps", "gamified learning", "adaptive learning", "edtech analytics", "AI in education", "digital literacy", "remote learning", "interactive whiteboards", "virtual labs", "student engagement", "education platforms", "digital assessments", "blended learning", "e-learning content", "microlearning", "STEM education", "education gamification", "AR/VR in education"
],
"Internet of Things (IoT)": [
    "IoT", "smart devices", "IoT platforms", "smart homes", "industrial IoT (IIoT)", "IoT sensors", "IoT security", "IoT analytics", "IoT protocols", "wearable IoT", "connected cars", "IoT networks", "IoT applications", "IoT in healthcare", "IoT in agriculture", "IoT in logistics", "smart cities", "IoT gateways", "IoT standards", "IoT data", "IoT cloud", "IoT edge computing"
],
"Game Development": [
    "game development", "game design", "Unity", "Unreal Engine", "game engines", "2D games", "3D games", "game programming", "game assets", "game physics", "game AI", "multiplayer games", "mobile games", "console games", "PC games", "indie game development", "game marketing", "VR games", "AR games", "game storytelling", "game monetization", "game testing", "game animation"
],
"Digital Marketing": [
    "digital marketing", "SEO", "content marketing", "social media marketing", "email marketing", "PPC (Pay Per Click)", "Google Ads", "Facebook Ads", "marketing automation", "affiliate marketing", "influencer marketing", "brand strategy", "online advertising", "analytics tools", "conversion optimization", "digital branding", "video marketing", "e-commerce marketing", "B2B marketing", "B2C marketing", "mobile marketing", "marketing trends"
],
"Web Development": [
    "web development", "frontend development", "backend development", "HTML", "CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Ruby on Rails", "PHP", "ASP.NET", "web APIs", "RESTful services", "GraphQL", "responsive design", "progressive web apps", "web hosting", "web performance optimization", "cross-browser compatibility", "web frameworks","web app","web","react","angular","vue","js","hypertext"
,"cascade","stylesheet","ui","ux"],
"Mobile App Development": [
    "app","applicat","mobile app development", "iOS apps", "Android apps", "Flutter", "React Native", "Swift", "Kotlin", "mobile app design", "cross-platform apps", "mobile app testing", "mobile app monetization", "mobile app analytics", "native apps", "hybrid apps", "mobile app marketing", "mobile app security", "mobile app stores", "mobile UX/UI", "mobile app deployment", "mobile backend"
],
"Environmental Technology": [
    "clean energy", "renewable energy", "solar power", "wind energy", "sustainable technology", "energy efficiency", "electric vehicles", "smart grids", "waste management", "water purification", "carbon capture", "green tech", "eco-friendly technology", "environmental monitoring", "climate tech", "biodiversity tech", "environmental analytics", "energy storage", "sustainable agriculture", "environmental sensors"
,"energ"],
  "Artificial Intelligence (AI) and Machine Learning (ML)": [
    "generative AI", "ChatGPT", "large language models", "AI personalization", "AI assistants", "AI ethics", "explainable AI", "AI for automation", "AI in healthcare", "AI in finance", "computer vision", "NLP (Natural Language Processing)", "AI in cybersecurity", "AI in climate modeling", "AI in gaming", "AI-driven analytics", "AI-powered robotics", "AI-based recommendation systems"
  ],
  "Blockchain and Cryptocurrency": [
    "DeFi (Decentralized Finance)", "Web3", "blockchain scalability", "NFTs", "crypto wallets", "smart contracts", "Ethereum 2.0", "blockchain interoperability", "layer 2 solutions", "DAO (Decentralized Autonomous Organizations)", "tokenization", "blockchain in supply chain", "CBDCs (Central Bank Digital Currencies)", "blockchain in gaming", "blockchain security"
  ],
  "Cloud Computing": [
    "cloud-native applications", "multi-cloud strategies", "serverless computing", "edge computing", "cloud AI services", "cloud automation", "hybrid cloud", "cloud analytics", "cloud-based DevOps", "cloud observability", "container orchestration", "Kubernetes", "cloud cost optimization", "cloud for IoT", "cloud gaming"
  ],
  "Cybersecurity": [
    "zero trust security", "endpoint detection", "XDR (Extended Detection and Response)", "cybersecurity automation", "threat intelligence platforms", "ransomware protection", "cloud security", "IoT security", "phishing defense", "AI in cybersecurity", "cyber resilience", "quantum-safe cryptography", "data breach prevention", "identity management", "SOC (Security Operations Center)"
  ],
  "Internet of Things (IoT)": [
    "smart cities", "industrial IoT (IIoT)", "IoT in healthcare", "IoT security", "edge computing for IoT", "IoT analytics", "smart homes", "connected vehicles", "wearable IoT", "IoT gateways", "IoT in agriculture", "IoT in logistics", "IoT in manufacturing", "IoT in energy", "IoT for predictive maintenance"
  ],
  "Artificial Reality (AR), Virtual Reality (VR), and Mixed Reality (MR)": [
    "Augumented reality","reality","Augumented","unity","unreal engine","blender","oculus","meta quest","Metaverse", "AR/VR gaming", "AR in retail", "VR for training and simulation", "AR in healthcare", "VR in education", "spatial computing", "AR in marketing", "haptics", "immersive content creation", "AR glasses", "VR headsets", "AR/VR collaboration tools", "XR development", "AR overlays for industrial use"
  ],
  "Autonomous Vehicles and Drones": [
    "self-driving cars", "autonomous delivery drones", "robotaxis", "autonomous trucking", "AI for autonomous systems", "V2X communication", "autonomous fleet management", "autonomous navigation", "LiDAR technology", "autonomous robotics", "drone swarms", "autonomous vehicle safety", "autonomous agricultural equipment", "smart traffic management"
  ],
  "Green and Renewable Energy Technologies": [
    "solar technology", "wind energy", "energy storage", "EV infrastructure", "clean energy solutions", "smart grids", "sustainable energy", "green hydrogen", "energy analytics", "carbon capture and storage", "microgrids", "energy efficiency", "renewable energy integration", "climate tech"
  ],
  "Biotechnology and Genomics": [
    "gene editing", "CRISPR technology", "synthetic biology", "genomic data analytics", "personalized medicine", "biopharmaceuticals", "biometrics", "healthcare genomics", "genome sequencing", "biotech in agriculture", "biotechnology innovation", "biotech startups", "DNA data storage", "genomic vaccines"
  ],
  "Quantum Computing": [
    "quantum AI", "quantum cryptography", "quantum error correction", "quantum simulation", "quantum optimization", "quantum machine learning", "quantum cloud computing", "quantum networking", "quantum programming languages", "quantum hardware", "quantum computing startups", "quantum algorithms", "quantum key distribution", "quantum supremacy"
  ],
  "Education Technology (EdTech)": [
    "AI in education", "gamified learning", "virtual classrooms", "adaptive learning", "e-learning platforms", "education apps", "remote learning tools", "VR in education", "AI tutors", "MOOCs", "blended learning", "education analytics", "digital assessments", "edtech gamification", "STEM education technologies"
  ],
  "Healthcare Technology": [
    "telemedicine", "remote patient monitoring", "AI in diagnostics", "wearable health devices", "digital therapeutics", "personalized healthcare", "smart hospitals", "healthcare robotics", "AI for drug discovery", "medical imaging technology", "blockchain for healthcare", "health data analytics", "biosensors", "mobile health (mHealth)"
  ]

    }
def classify_domain(idea):
    idea_lower = idea.lower()
    domain_set = set()

    # Step 1: Pre-filter classification based on domain-specific keywords
    domain_set = set()
    for domain, keywords in domain_keywords.items():
        if any(keyword in idea_lower for keyword in keywords):
            domain_set.add(domain)

# Step 2: Use NLP classification pipeline to generate potential domains
    domain_classification = nlp(idea_lower, candidate_labels=[
    "AI", "Agriculture", "IoT", "ML", "Clean and Green Technology", "Environment", "Blockchain",
    "Cybersecurity", "Robotics", "Data Science", "Healthcare Technology", "Autonomous Vehicles","Education Technology (EdTech)","Quantum Computing","Biotechnology and Genomics",
    "Green and Renewable Energy Technologies","Autonomous Vehicles and Drones","Artificial Reality (AR)/Virtual Reality (VR)/and Mixed Reality (MR)",
    "Cybersecurity","Cloud Computing"])

# Step 3: Add the domains from the NLP step (only domains with score > 0.9)
    nlp_domains = set()
    for domain, score in zip(domain_classification['labels'], domain_classification['scores']):
        if score > 0.065:  # Only add domains with a score higher than a threshold (e.g., 0.5)
            nlp_domains.add(domain)

# Step 4: Find common domains between the two sets (domain_set and nlp_domains)
    common_domains = domain_set.intersection(nlp_domains)
    if not common_domains:
        common_domains = domain_set
        

    return list(common_domains)  # Convert set to list for output