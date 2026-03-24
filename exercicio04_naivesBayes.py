from collections import defaultdict

class NaiveBayes:
    def __init__(self):
        self.class_counts = defaultdict(int)
        self.feature_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.feature_values = defaultdict(set)
        self.total_samples = 0
        self.classes = set()

    def fit(self, X, y):
        self.total_samples = len(y)

        for i in range(len(X)):
            label = y[i]
            self.class_counts[label] += 1
            self.classes.add(label)

            for feature, value in X[i].items():
                self.feature_counts[label][feature][value] += 1
                self.feature_values[feature].add(value)

    def predict_proba(self, x):
        probs = {}
        k = len(self.classes)

        for c in self.classes:
            # Prior com Laplace
            prior = (self.class_counts[c] + 1) / (self.total_samples + k)

            likelihood = 1

            for feature, value in x.items():
                count = self.feature_counts[c][feature][value]
                total = sum(self.feature_counts[c][feature].values())
                num_values = len(self.feature_values[feature])

                # Laplace smoothing
                prob = (count + 1) / (total + num_values)
                likelihood *= prob

            probs[c] = prior * likelihood

        # Normalização
        total_prob = sum(probs.values())
        for c in probs:
            probs[c] /= total_prob

        return probs

    def predict(self, x):
        probs = self.predict_proba(x)
        return max(probs, key=probs.get)


# ===============================
# 📊 Dataset (Lentes de Contato)
# ===============================

X = [
    {"idade": "infantil", "diagnostico": "miopia", "astigmatismo": "não", "lacrimal": "reduzida"},
    {"idade": "infantil", "diagnostico": "miopia", "astigmatismo": "não", "lacrimal": "normal"},
    {"idade": "infantil", "diagnostico": "hipermetropia", "astigmatismo": "não", "lacrimal": "reduzida"},
    {"idade": "infantil", "diagnostico": "hipermetropia", "astigmatismo": "não", "lacrimal": "normal"},
    {"idade": "adolescente", "diagnostico": "miopia", "astigmatismo": "sim", "lacrimal": "normal"},
    {"idade": "adolescente", "diagnostico": "miopia", "astigmatismo": "sim", "lacrimal": "reduzida"},
    {"idade": "adolescente", "diagnostico": "hipermetropia", "astigmatismo": "sim", "lacrimal": "normal"},
    {"idade": "adolescente", "diagnostico": "hipermetropia", "astigmatismo": "sim", "lacrimal": "reduzida"},
    {"idade": "adulto", "diagnostico": "miopia", "astigmatismo": "sim", "lacrimal": "normal"},
    {"idade": "adulto", "diagnostico": "miopia", "astigmatismo": "sim", "lacrimal": "reduzida"},
    {"idade": "adulto", "diagnostico": "hipermetropia", "astigmatismo": "sim", "lacrimal": "normal"},
    {"idade": "adulto", "diagnostico": "hipermetropia", "astigmatismo": "sim", "lacrimal": "reduzida"},
]

y = [
    "nenhuma",
    "gelatinosa",
    "nenhuma",
    "gelatinosa",
    "dura",
    "nenhuma",
    "dura",
    "nenhuma",
    "dura",
    "nenhuma",
    "dura",
    "nenhuma",
]

# ===============================
# 🧪 Treinamento
# ===============================

model = NaiveBayes()
model.fit(X, y)

# ===============================
# 🔍 Teste (caso do exercício)
# ===============================

entrada = {
    "idade": "infantil",
    "diagnostico": "hipermetropia",
    "astigmatismo": "não",
    "lacrimal": "reduzida"
}

probs = model.predict_proba(entrada)
pred = model.predict(entrada)

print("Probabilidades:")
for classe, prob in probs.items():
    print(f"{classe}: {prob:.4f}")

print("\nClasse prevista:", pred)