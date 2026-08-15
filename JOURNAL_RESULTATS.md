# Journal de résultats — TER Hallucination Detection

Ce fichier centralise TOUS les résultats bruts et analyses critiques au fur
et à mesure des runs.
À tenir à jour après chaque run, pour ne jamais perdre un résultat.

---

## Mistral-7B-Instruct-v0.3 (TERMINÉ — 5/5 pipelines)

### Self-Consistency (TruthfulQA, N=30, N_gen=10-20 selon le run)
- Precision=0.50, Recall=0.15, F1=0.23
- CSV: results/self_consistency_mistral-7b_truthfulqa.csv

### Confidence Calibration (TruthfulQA, N=30)
- Spearman rho = -0.055 (p=0.77, NON significatif)
- Sur-confiance moyenne = +43.5 points
- CSV: results/calibration_mistral-7b_truthfulqa.csv
- **Critique**: bien plus marqué que le rapport 2 (rho 0.41-0.53 attendu),
  probablement dû au petit échantillon (N=30 vs 817).

### RAG (TruthfulQA, N=30)
- Hallucination sans RAG: 43.3% | avec RAG: 73.3% (RAG DÉGRADE de +30pp)
- Latence RAG: +36.27s
- CSV: results/rag_mistral-7b_truthfulqa.csv
- **Critique**: corpus RAG (212 passages, 10 sujets ciblés) mal aligné avec
  la diversité des 38 catégories TruthfulQA → retrieval non pertinent qui
  nuit plutôt qu'aide. Résultat contre-intuitif mais scientifiquement
  défendable et à documenter tel quel.

### Prompting Analysis (TruthfulQA, N=20 par stratégie)
- Zero-shot: 65.0% accuracy, SC-Score moyen 0.74
- Few-shot (3 ex.): 60.0% accuracy, SC-Score moyen 0.69
- Chain-of-Thought: 60.0% accuracy, SC-Score moyen 0.68
- CSV: results/prompting_analysis_mistral-7b.csv
- **Critique**: inverse du rapport 2 (qui attendait CoT meilleur que
  zero-shot de -14pts hallucination). Possible sur-réflexion du CoT sur
  questions factuelles simples, ou effet d'échantillon réduit (N=20 vs 200).

### Factual Scoring Tool — 3 démonstrations
1. Marie Curie (hors corpus) → score 1.00 mais erreur Chemistry 1912 (vrai:
   1911) validée à tort. **Critique**: sujet absent du corpus RAG → retrieval
   non pertinent → faux positif.
2. Einstein né 1879, Nobel Physique **1925** (vrai: 1921) → score 1.00,
   erreur numérique validée à tort. **Critique**: les modèles NLI sont peu
   sensibles aux erreurs numériques isolées dans une phrase par ailleurs
   correcte (limite connue de l'approche NLI pour la vérification factuelle).
3. Einstein "born in France" (faux, erreur qualitative) → l'extraction a
   remplacé par "[Location not specified]", jamais vérifié. **Critique**:
   utiliser le même LLM pour extraire ET vérifier peut faire "disparaître"
   silencieusement une erreur dès l'étape d'extraction, avant même la
   vérification NLI — biais méthodologique à documenter.

---

## Gemma-2-9B-it (EN COURS)

### Self-Consistency (TruthfulQA, N=30, N_gen=10)
- **Precision=0.667, Recall=0.500, F1=0.571**
- CSV: results/self_consistency_gemma-2-9b_truthfulqa.csv
- **Comparaison Mistral vs Gemma**: Gemma nettement supérieur (F1=0.57 vs
  0.23 pour Mistral) — cohérent avec le constat du rapport 2 ("Gemma-2-9B
  domine légèrement sur les deux benchmarks"). Bon point de discussion:
  différence marquée même sur petit échantillon, architecture/entraînement
  plus récent de Gemma semble donner une meilleure auto-cohérence.

### Confidence Calibration (TruthfulQA, N=30)
- **Spearman rho = 0.252 (p=0.18, non significatif à 0.05 mais direction correcte)**
- **Sur-confiance moyenne = +39.7 points**
- CSV: results/calibration_gemma-2-9b_truthfulqa.csv
- **Comparaison Mistral vs Gemma**: Gemma a une meilleure corrélation
  confiance/exactitude que Mistral (rho=0.25 vs -0.055) — cohérent avec le
  meilleur F1 self-consistency observé. Sur-confiance élevée dans les deux
  cas (+39.7 vs +43.5 pts), donc les deux modèles restent globalement
  sur-confiants, mais Gemma "sait un peu mieux ce qu'il ne sait pas".

### RAG (TruthfulQA, N=30)
- Hallucination sans RAG: 36.7% | avec RAG: 73.3% (RAG DÉGRADE de +36.7pp)
- Latence RAG: +31.20s
- CSV: results/rag_gemma-2-9b_truthfulqa.csv
- **Critique renforcée**: le RAG dégrade la performance sur LES DEUX modèles
  testés jusqu'ici (Mistral +30pp, Gemma +36.7pp) — ce n'est donc PAS un cas
  isolé/bruit d'un seul modèle, mais un effet systématique du corpus mal
  aligné avec la diversité thématique de TruthfulQA (38 catégories vs 10
  sujets couverts par le corpus RAG). C'est un résultat robuste et un point
  fort de discussion : la conclusion "RAG réduit les hallucinations" du
  rapport 2 (Table 3, -13pp) ne se généralise pas à un corpus mal ciblé —
  la pertinence du corpus est une condition nécessaire, pas un détail.

### Prompting Analysis (TruthfulQA, N=20 par stratégie)
- Zero-shot: 55.0% accuracy, SC-Score moyen 0.51
- Few-shot (3 ex.): 65.0% accuracy, SC-Score moyen 0.50
- Chain-of-Thought: 45.0% accuracy, SC-Score moyen 0.39
- CSV: results/prompting_analysis_gemma-2-9b.csv
- **Critique renforcée (pattern sur 2 modèles)**: Chain-of-Thought est la
  stratégie la MOINS performante pour Mistral (60%, ex-aequo) ET pour Gemma
  (45%, nettement la pire) — contredit systématiquement l'hypothèse du
  rapport 2 (CoT devait réduire l'hallucination de -14pts). C'est un
  résultat robuste et contre-intuitif à assumer et discuter : sur des
  questions factuelles courtes de type TruthfulQA (pas des problèmes de
  raisonnement multi-étapes), inciter le modèle à "réfléchir étape par
  étape" semble l'amener à sur-analyser ou dériver, augmentant le risque
  d'hallucination plutôt que de le réduire. Meilleure stratégie variable
  selon le modèle (zero-shot pour Mistral, few-shot pour Gemma) — pas de
  stratégie universellement optimale sur cet échantillon réduit.

### Factual Scoring Tool — Marie Curie (même corpus, hors-sujet)
- Score 4/4 = 1.00, mais erreur Chemistry 1912 (vrai: 1911) validée à tort
  — identique au comportement observé sur Mistral.
- **Confirme que le biais vient du CORPUS (Marie Curie absente), pas du
  modèle** : reproductible indépendamment du LLM utilisé pour l'extraction
  et la vérification. Renforce la robustesse de cette critique pour le
  rapport.

**GEMMA-2-9B: 5/5 PIPELINES TERMINÉS ✅**

---

## LLaMA-3-8B-Instruct (EN COURS)

### Self-Consistency (TruthfulQA, N=30, N_gen=10)
- **Precision=0.455, Recall=0.333, F1=0.385**
- CSV: results/self_consistency_llama-3-8b_truthfulqa.csv
- **Comparaison 3 modèles (self-consistency, F1)**: Gemma-2-9B (0.571) >
  LLaMA-3-8B (0.385) > Mistral-7B (0.231). Classement cohérent avec le
  rapport 2 qui plaçait déjà Gemma légèrement devant. LLaMA-3 se positionne
  entre les deux, comme attendu pour un modèle de taille intermédiaire.

### Confidence Calibration (TruthfulQA, N=30, mais seulement 13/30 confiances valides)
- **Spearman rho = 0.244 (p=0.47, non significatif)**
- **Sur-confiance moyenne = +41.8 points** (calculée sur les valeurs valides)
- CSV: results/calibration_llama-3-8b_truthfulqa.csv
- **Critique importante — taux d'échec de parsing élevé**: 17/30 réponses
  ont `conf=-1` (échec d'extraction d'un entier depuis la réponse du
  modèle), contre quasiment 0 échec pour Mistral et Gemma sur le même
  prompt exact. LLaMA-3 semble répondre au prompt de confiance verbalisée
  de façon moins disciplinée (texte au lieu d'un chiffre seul, refus,
  reformulation...). Conséquence: le rho de LLaMA-3 est calculé sur un
  échantillon effectif de 13 points seulement, pas 30 — à mentionner
  explicitement dans le rapport comme limite de la méthode d'extraction
  (regex sur chiffres) et piste d'amélioration (ex: contraindre la
  génération avec un format structuré / grammar constrained decoding).

### RAG (TruthfulQA, N=30)
- Hallucination sans RAG: 63.3% | avec RAG: 73.3% (RAG DÉGRADE de +10.0pp)
- Latence RAG: +27.09s
- CSV: results/rag_llama-3-8b_truthfulqa.csv
- **CONFIRMATION SUR 3/3 MODÈLES**: Mistral (+30.0pp), Gemma (+36.7pp),
  LLaMA-3 (+10.0pp) — le RAG dégrade systématiquement la performance sur
  les TROIS modèles testés, avec le corpus ciblé actuel. C'est le résultat
  le plus robuste et le plus fort de tout le projet : reproductible
  indépendamment du modèle, donc clairement imputable au corpus/pipeline
  RAG et non à un modèle en particulier. Point central à mettre en avant
  dans la Discussion et probablement en ouverture de la présentation orale
  — résultat contre-intuitif, bien expliqué, qui démontre un vrai esprit
  critique plutôt qu'une simple reproduction de résultats attendus.
  Note: LLaMA-3 a aussi le taux d'hallucination SANS RAG le plus élevé
  des 3 modèles (63.3% vs 43.3% Mistral, 36.7% Gemma) — cohérent avec son
  F1 self-consistency intermédiaire.

### Prompting Analysis (TruthfulQA, N=20 par stratégie)
- Zero-shot: 50.0% accuracy, SC-Score moyen 0.61
- Few-shot (3 ex.): 45.0% accuracy, SC-Score moyen 0.55
- Chain-of-Thought: 50.0% accuracy, SC-Score moyen 0.63
- CSV: results/prompting_analysis_llama-3-8b.csv
- **Nuance importante — le pattern ne se généralise PAS aux 3 modèles**:
  contrairement à Mistral (CoT pire, -5pts vs zero-shot) et Gemma (CoT très
  nettement pire, -10pts vs zero-shot ET pire que few-shot), pour LLaMA-3
  le CoT est à ÉGALITÉ avec zero-shot (50% chacun) et meilleur que le
  few-shot (45%, le pire ici). À documenter honnêtement dans le rapport:
  la meilleure stratégie de prompting dépend du modèle, sans règle
  universelle observée sur cet échantillon réduit. Le seul pattern robuste
  est l'ABSENCE de bénéfice net et systématique du CoT (jamais clairement
  meilleur que zero-shot sur aucun des 3 modèles), ce qui contredit déjà
  l'hypothèse initiale du rapport 2, mais sans qu'on puisse dire que CoT
  est systématiquement nuisible non plus (cas LLaMA-3). Nuance à assumer
  plutôt qu'à lisser artificiellement.

### Factual Scoring Tool — Marie Curie (même corpus, hors-sujet)
- Score 6/6 = 1.00 (LLaMA-3 a découpé en 6 claims au lieu de 4, plus
  granulaire), mais erreur Chemistry 1912 (vrai: 1911) validée à tort —
  identique à Mistral et Gemma.
- **Confirmé sur 3/3 modèles** : le biais "corpus hors-sujet → validation
  erronée" est totalement indépendant du LLM utilisé. C'est un résultat
  très robuste pour le rapport.

**LLAMA-3-8B: 5/5 PIPELINES TERMINÉS ✅**

---

# 🎉 LES 3 MODÈLES SONT COMPLETS — 15/15 PIPELINES TERMINÉS 🎉

Prochaine étape : agrégation finale (generate_report_figures.py) + tests
de significativité statistique, puis rédaction complète du rapport.

## Test complémentaire N=100 (10 août) — vérification de la robustesse

### Mistral-7B, self-consistency, N=100 (au lieu de N=30)
- **Precision=0.455, Recall=0.213, F1=0.290**
- Comparaison N=30 vs N=100 : F1 passe de 0.231 à 0.290 — reste bien dans
  l'intervalle de confiance bootstrap calculé à N=30 [0.000, 0.444].
  Cohérence rassurante : le résultat ne bouge pas de façon aberrante avec
  un échantillon plus large, juste un léger ajustement à la hausse.
- CSV: results/self_consistency_mistral-7b_truthfulqa.csv (ÉCRASÉ,
  contient maintenant N=100 -- sauvegarder une copie séparée si besoin de
  garder aussi la version N=30 pour archive)

### Gemma-2-9B, self-consistency, N=100
- **Precision=0.528, Recall=0.339, F1=0.413**
- Comparaison N=30 vs N=100 : F1 passe de 0.571 à 0.413 — reste dans
  l'intervalle de confiance bootstrap calculé à N=30 [0.320, 0.774], mais
  ajustement plus marqué à la baisse que pour Mistral. L'écart Mistral/
  Gemma se resserre nettement à N=100 (0.290 vs 0.413, delta=0.123) par
  rapport à N=30 (0.231 vs 0.571, delta=0.340) — cohérent avec l'hypothèse
  que l'écart à N=30 était en partie gonflé par le bruit d'échantillonnage.
  À confirmer avec McNemar sur ces données N=100.
- CSV: results/self_consistency_gemma-2-9b_truthfulqa.csv (ÉCRASÉ, N=100)

### Décision méthodologique — arrêt du complément après perte de session (15 août)

La session Colab s'est déconnectée avant que le CSV détaillé de Gemma N=100
ait pu être téléchargé, empêchant le calcul du test McNemar apparié exact
sur N=100 (qui nécessite le détail question-par-question pour les deux
modèles sur le même run). Décision assumée : ne pas relancer un 3e run
(~1h30-2h supplémentaires) vu le calendrier serré (rendu le 20 août), et
documenter honnêtement cette limite dans le rapport plutôt que de
poursuivre indéfiniment la vérification statistique.

**Ce qu'on peut conclure quand même, avec les estimations ponctuelles
disponibles** : le resserrement de l'écart de F1 entre N=30 (0.340) et
N=100 (0.123) est cohérent avec l'explication la plus probable -- une
partie de l'écart observé à N=30 relevait du bruit d'échantillonnage, et
un échantillon plus large tend à rapprocher les deux modèles plutôt qu'à
confirmer une différence tranchée. Cela renforce, plutôt que ça
n'affaiblit, la conclusion prudente déjà écrite dans le rapport ("tendance
observée, non établie statistiquement à N=30") -- le nouvel échantillon
va dans le même sens de prudence.
- ⚠️ Note technique : le chemin affiché montre un dossier dupliqué
  (TER-hallucination-detection/TER-hallucination-detection/results/...)
  car le clone git a été relancé dans un dossier déjà existant. Vérifier
  le bon chemin avant de télécharger le CSV.

### Comparaison N=100 : Mistral (F1=0.290) vs Gemma (F1=0.455)
Écart de F1 réduit par rapport à N=30 (0.231 vs 0.571, écart de 0.34) à
N=100 (0.290 vs 0.455, écart de 0.165) -- l'écart se resserre avec un
échantillon plus grand. Test McNemar à faire pour confirmer si cet écart
plus modeste devient significatif ou reste non significatif.

## Tests de significativité statistique (self-consistency, N=30)

### Bootstrap CI (95%, 2000 rééchantillonnages)
- Mistral-7B: F1=0.231, IC95%=[0.000, 0.444]
- LLaMA-3-8B: F1=0.385, IC95%=[0.111, 0.606]
- Gemma-2-9B: F1=0.571, IC95%=[0.320, 0.774]

### McNemar (comparaison 2 à 2)
- Mistral vs LLaMA-3: discordants=11, p=1.0000 → NON significatif
- Mistral vs Gemma: discordants=14, p=0.1796 → NON significatif
- LLaMA-3 vs Gemma: discordants=17, p=0.6291 → NON significatif

**CRITIQUE IMPORTANTE ET HONNÊTE À NE PAS OMETTRE**: malgré un écart de F1
apparemment important entre Mistral (0.23) et Gemma (0.57) — qui semblait
raconter une histoire claire de "Gemma domine" —, AUCUNE des comparaisons
par paires n'est statistiquement significative à N=30 (tous les p-values
> 0.05, le plus proche étant Mistral vs Gemma à p=0.18). Les intervalles
de confiance bootstrap se chevauchent également de façon importante
(Mistral [0, 0.44] et Gemma [0.32, 0.77] se chevauchent légèrement autour
de 0.32-0.44 ; LLaMA-3 chevauche complètement les deux autres).

**Implication pour le rapport**: on ne peut PAS affirmer avec une
confiance statistique établie que Gemma "est meilleur" que Mistral sur la
tâche de détection par self-consistency — seulement qu'on OBSERVE cette
tendance sur cet échantillon de 30 questions, cohérente avec le rapport 2
(qui utilisait n=817, donc un pouvoir statistique bien supérieur). C'est
une limite assumée du calendrier compressé (§3.1), pas une invalidation
du résultat — mais il faut le formuler avec la nuance appropriée dans la
Discussion : "tendance observée" plutôt que "démontré". C'est exactement
le genre de rigueur qui distingue un bon rapport d'un rapport qui
sur-interprète des différences non significatives.

En agrégeant les résultats (results/aggregated_tables.md), le taux
d'hallucination AVEC RAG est exactement 73.3% pour Mistral, LLaMA-3 ET
Gemma — une coïncidence trop exacte pour être due au hasard (indépendance
entre 3 modèles différents donnerait des taux légèrement différents).

**Explication la plus probable** : le corpus FAISS (212 passages, 10
sujets fixes) est identique pour les 3 modèles. Pour une bonne partie des
30 questions TruthfulQA, aucun passage pertinent n'est récupéré — et le
prompt RAG demande explicitement au modèle de dire "si les passages ne
contiennent pas la réponse" quand c'est le cas. Il est probable qu'un
sous-ensemble FIXE de questions (déterminé par le corpus, pas par le
modèle) produise systématiquement des réponses de type "je ne peux pas
déterminer la réponse", qui échouent ensuite au test de correspondance
lexicale (is_answer_correct) quel que soit le modèle — d'où un taux
similaire/identique indépendamment du LLM utilisé.

**CONFIRMÉ EMPIRIQUEMENT** (script check_rag_overlap.py) : sur les 22
questions hallucinées "avec RAG" par modèle, **19 sont EXACTEMENT LES
MÊMES pour les 3 modèles** (mistral-7b, llama-3-8b, gemma-2-9b) —
indices [2,3,5,6,8,10,11,12,15,18,20,21,22,23,24,26,27,28,29]. Ce n'est
donc PAS une coïncidence numérique : c'est bien le CORPUS/RETRIEVAL qui
détermine, indépendamment du modèle génératif, quelles questions posent
problème avec RAG. Preuve solide et quantifiée pour le rapport : le
résultat "RAG dégrade la performance" n'est pas un artefact d'un modèle
en particulier, mais un effet structurel du pipeline retrieval + corpus
mal aligné avec la diversité de TruthfulQA. C'est l'argument le plus
rigoureux du projet — appuyé par une vérification empirique directe, pas
juste une intuition.

---

## Incident technique — quota Colab gratuit épuisé (9 août)

Le run self-consistency LLaMA-3 a été interrompu à 11/30 par une
déconnexion runtime liée à l'épuisement du quota GPU gratuit journalier de
Colab ("Cannot connect to GPU backend due to usage limits"). Passage à
**Colab Pro** (abonnement payant) pour débloquer et sécuriser la suite des
runs — session relancée proprement avec succès derrière, run complet
achevé sans coupure. À mentionner dans le rapport (section limitations
pratiques / contraintes de calendrier et de ressources) si pertinent.

---

## Décisions méthodologiques prises en cours de route (à justifier dans le rapport)

1. N=20→N=10 générations self-consistency (contrainte calendrier, déjà
   justifié dans rapport 2 §7 "Challenges Encountered").
2. Échantillons réduits (--limit 30 au lieu de 817/5000) pour tous les
   benchmarks, compensé par tests de significativité (bootstrap, McNemar).
3. Corpus RAG = échantillon ciblé Wikipedia (212 passages, 10 sujets),
   pas le dump complet 20M passages — limite documentée explicitement.
4. Fix technique: dataset "truthful_qa" → "truthfulqa/truthful_qa"
   (namespace requis par versions récentes de huggingface_hub).
5. Fix technique: lib "wikipedia" (Python) remplacée par appel direct API
   MediaWiki avec User-Agent (bug de fiabilité, 0-4/10 sujets réussis avant
   le fix, 10/10 après).
6. LLaMA-3-8B: accès officiel Meta validé après ~24h d'attente (pas eu
   besoin du miroir NousResearch de secours).

---

## Statut final du projet (15 août)

- [x] Terminer Gemma-2-9B (5 pipelines)
- [x] Terminer LLaMA-3-8B (5 pipelines)
- [ ] HaluEval — non fait, documenté comme limitation assumée dans le rapport (§6.2)
- [x] Lancer generate_report_figures.py une fois tous les modèles faits
- [x] Lancer les tests de significativité (McNemar, bootstrap), + vérification complémentaire N=100
- [x] Remplir toutes les sections "À COMPLÉTER" du rapport final LaTeX
- [x] Relecture, mise en forme, vérification cohérence chiffres
- [ ] Préparer les slides (deadline: 24h avant soutenance du 26/27 août — pas encore commencé, pas urgent)

**Rapport final : `rapport_final_TER.pdf`, 16 pages, terminé et vérifié.**
**Repository GitLab : à jour avec tous les résultats et le code.**
