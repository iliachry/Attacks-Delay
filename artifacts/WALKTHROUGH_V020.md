# Queueing Networks Under Adversarial Attacks & Timeouts (`aqnet v0.2.0` — Saddle-Node Retry Bifurcation & Studio)

> [!IMPORTANT]
> **Triple-9+ Goldwood Index (`GWI v1.4.0`) Πιστοποίηση**:
> - **Profile `PAPER`**: **$\mathcal{G}_{\text{paper}} = 9.62 / 10.00$ `PASS`** (`0` blockers, `0` Overfull boxes, `0` Type-3 fonts, `0` `\resizebox` tables)
> - **Profile `APP`**: **$\mathcal{G}_{\text{app}} = 9.64 / 10.00$ `PASS`** (`0` blockers, $\mathcal{R}_{\text{kanon}} = 1.0000$, Law `L8` $15 \le 24$ controls, $\mathcal{S}_{L8} = 10.0$)
> - **Profile `VIDEO`**: **$\mathcal{G}_{\text{video}} = 9.50 / 10.00$ `PASS`** (`0` blockers, `1920x1080 @ 30fps`, `1.48 MB <= 20 MB`, EBU R128 `I = -13.7 LUFS`, `TP = -2.3 dBTP`, `LRA = 2.0 LU`)

---

## 1. Βίντεο Περιήγησης 1080p (`< 20 MB` Companion Edition — `GWI-VID-P01` & `GWI-VID-C01`)

![Walkthrough Video — Adversarial Queueing & Saddle-Node Bifurcation Studio](/Users/iliaschrysovergis/.gemini/antigravity/brain/015b8f5a-90aa-4039-a429-3a1553b7e25f/walkthrough_compressed.mp4)

````carousel
![Publication Figure 5 — Saddle-Node Fold Diagram & Phase-Space Portrait](/Users/iliaschrysovergis/.gemini/antigravity/brain/015b8f5a-90aa-4039-a429-3a1553b7e25f/fig_saddle_node_bifurcation.png)
<!-- slide -->
![9-Frame Walkthrough Contact Sheet](/Users/iliaschrysovergis/.gemini/antigravity/brain/015b8f5a-90aa-4039-a429-3a1553b7e25f/walkthrough_contact_sheet.png)
<!-- slide -->
![3-Viewport Responsive Verification (Desktop 1440×900 · Tablet 768×1024 · Mobile 390×844)](/Users/iliaschrysovergis/.gemini/antigravity/brain/015b8f5a-90aa-4039-a429-3a1553b7e25f/3viewport_responsive_sheet.png)
````

---

## 2. Track A — Μαθηματικός Πυρήνας `aqnet v0.2.0` (`SymPy` + `SimPy` + CLI)

Στο [bifurcation.py](file:///Volumes/externalMac/GitHub/Attacks-Delay/src/aqnet/models/bifurcation.py) και [cli.py](file:///Volumes/externalMac/GitHub/Attacks-Delay/src/aqnet/cli.py) υλοποιήθηκε η πλήρης αναλυτική θεωρία της **Διακλάδωσης Σέλας-Κόμβου (Saddle-Node / Fold Bifurcation)** για δίκτυα ουρών αναμονής υπό επιθέσεις και επανεκπομπές λόγω λήξης χρονικού ορίου ($T$):

1. **Κλειστή Λύση μέσω της Συνάρτησης Wright-$\omega$ / Lambert-$W_0$ (Case 1 — Destruction & Case 2 — Modification)**:
   - Για ουρά $M/M/1$ με ενεργό ρυθμό εκκαθάρισης $\gamma = \mu - \lambda_{\text{eff}}$, ο συνολικός χρόνος παραμονής $W_q + S \sim \text{Exp}(\gamma)$ έχει πιθανότητα επιτυχίας εντός του χρονικού ορίου $T$: $P_{\text{succ}} = (1-p)\left(1 - e^{-\gamma T}\right)$.
   - Η συνάρτηση πραγματικής διεκπεραίωσης (goodput) $\Gamma_{\text{mod}}(\Lambda^*) = (1-p)\Lambda^* \left[1 - e^{-(\mu - \Lambda^*)T}\right]$ παρουσιάζει σημείο αναδίπλωσης (fold) $\frac{\partial \Gamma_{\text{mod}}}{\partial \Lambda^*} = 0$ στο:
     $$\Lambda_{\text{SN}}^{\text{mod}}(\mu, T) = \frac{\omega(\mu T + 1) - 1}{T} = \frac{W_0(e^{\mu T + 1}) - 1}{T}$$
   - Η ακριβής κρίσιμη ικανότητα εξωγενούς κίνησης είναι:
     $$\lambda_{\text{crit}}^{\text{SN,mod}}(p, \mu, T) = \frac{1-p}{T}\frac{\left[\omega(\mu T + 1) - 1\right]^2}{\omega(\mu T + 1)}$$
   - Η κανονική μορφή τετραγωνικής ρίζας κοντά στο fold ($\lambda = \lambda_{\text{crit}}^{\text{SN}} - \Delta\lambda$) δίνει τους δύο κλάδους ισορροπίας:
     $$\Lambda_{\text{stable/unstable}}^*(\lambda) = \Lambda_{\text{SN}} \mp \sqrt{\frac{\Delta\lambda}{c_2}} + \mathcal{O}(\Delta\lambda), \qquad c_2^{\text{mod}} = -\frac{1}{2}\frac{\partial^2 \Gamma_{\text{mod}}}{\partial (\Lambda^*)^2}\Bigg|_{\Lambda_{\text{SN}}^{\text{mod}}} = \frac{(1-p)T\left[1 + \omega(\mu T + 1)\right]}{2\,\omega(\mu T + 1)} > 0$$
2. **Θεώρημα Δομικής Αμεταβλητότητας ως προς $p$ (Case 1 — Destruction vs. Case 2 — Modification)**:
   - Στο Case 1 (Destruction με προ-εξυπηρετητικό φιλτράρισμα), θέτοντας ενεργό ρυθμό εισόδου στον εξυπηρετητή $x = (1-p)\Lambda^*$, προκύπτει $\Gamma_{\text{dest}}(\Lambda^*) = x\left[1 - e^{-(\mu - x)T}\right]$. Επομένως, το κρίσιμο όριο $\lambda_{\text{crit}}^{\text{SN,dest}}(\mu, T) = \frac{[\omega(\mu T + 1)-1]^2}{T\,\omega(\mu T + 1)}$ είναι **αυστηρά ανεξάρτητο του $p \in [0, 1)$**, ενώ ο ρυθμός προσπαθειών στο fold κλιμακώνεται ως $\Lambda_{\text{SN}}^{\text{dest}}(p) = \frac{\omega(\mu T + 1)-1}{(1-p)T} \propto (1-p)^{-1}$ και η καμπυλότητα ως $c_2^{\text{dest}}(p) = (1-p)^2 c_2^{\text{dest}}(0)$.
   - Αντίθετα, στο Case 2 (Modification), τα αλλοιωμένα πακέτα καταναλώνουν πλήρη χρόνο εξυπηρέτησης πριν απορριφθούν, διαβρώνοντας γραμμικά την κρίσιμη χωρητικότητα κατά $(1-p)$.
3. **Συμβολική Επαλήθευση `SymPy`, Προσομοίωση `SimPy` & Ισοδυναμία 5 Τοπολογιών**:
   - Η συνάρτηση `verify_symbolic_bifurcation_proofs()` επαληθεύει συμβολικά 4 ταυτότητες (`SymPy` residual `0`).
   - Η συνάρτηση `simulate_metastable_retry_collapse_simpy()` προσομοιώνει την απόδραση μιας πεπερασμένης ουράς πέρα από το ασταθές διαχωριστικό όριο (separatrix) $\Lambda_{\text{unstable}}^*(\lambda)$ όταν $\lambda \to \lambda_{\text{crit}}^{\text{SN}}$.
   - **Test Suite**: `24/24` τεστ στο [test_models_and_sim.py](file:///Volumes/externalMac/GitHub/Attacks-Delay/tests/test_models_and_sim.py) (`93%` κάλυψη κώδικα, `0` σφάλματα `ruff`, απόλυτη ισοδυναμία `< 1.5e-6` μεταξύ `aqnet.models.bifurcation`, `aqnet.models` και `web/app.js`).

| Τοπολογία ($\mu=10, p=0.25, T=1.0\text{ s}$) | Fold $\Lambda_{\text{SN}}^*$ | Κρίσιμο Όριο $\lambda_{\text{crit}}^{\text{SN}}$ | Αξιοποίηση $\rho_{\text{SN}}$ | Καμπυλότητα $c_2$ | Αφελές Όριο $\lambda_{\max}$ | Λόγος $\lambda_{\text{crit}}^{\text{SN}} / \lambda_{\max}$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Case 1: 1-Node Destruction** | $10.4302$ | **$6.9360$** ($p$-invariant) | $0.7823$ | $0.3131$ | $10.0000$ | $69.36\%$ |
| **Case 2: 1-Node Modification** | $7.8227$ | **$5.2020$** | $0.7823$ | $0.4175$ | $7.5000$ | $69.36\%$ |
| **Case 3: $N$-Node Tandem ($N=2$)** | $6.6788$ | **$3.1518$** | $0.6679$ | $0.2330$ | $5.6250$ | $56.03\%$ |
| **Case 4: $N$-Node Feedforward ($N=2$)** | $7.3034$ | **$3.4780$** | $0.7303$ | $0.2373$ | $5.6250$ | $61.83\%$ |
| **Case 5: $N$-Node Feedback ($N=2$)** | $7.6389$ | **$1.5656$** | $0.7639$ | $0.0967$ | $2.5000$ | $62.62\%$ |

---

## 3. Track B — Αναβάθμιση Χειρογράφων LaTeX & Διανυσματικών Σχημάτων (`GWI-PAP-P02` & `GWI-PAP-V01`)

- **[paper.tex](file:///Volumes/externalMac/GitHub/Attacks-Delay/paper.tex) & [paper.pdf](file:///Volumes/externalMac/GitHub/Attacks-Delay/paper.pdf) (19 σελίδες, Archival Preprint)**: Προστέθηκε η Ενότητα 8 (*Saddle-Node Retry Bifurcation and Critical Collapse Theory*, Θεώρημα 8.1, Πόρισμα 8.2, Σχήμα 5, Πίνακας 3) και εναρμονίστηκαν όλες οι εξισώσεις των 5 τοπολογιών.
- **[paper_ieee.tex](file:///Volumes/externalMac/GitHub/Attacks-Delay/paper_ieee.tex) & [paper_ieee.pdf](file:///Volumes/externalMac/GitHub/Attacks-Delay/paper_ieee.pdf) (7 σελίδες, IEEE Double-Column)**: Προστέθηκε η Ενότητα VIII και αφαιρέθηκαν όλα τα `\resizebox` γύρω από `tabular` (`GWI-PAP-P02`).
- **[letter/letter.tex](file:///Volumes/externalMac/GitHub/Attacks-Delay/letter/letter.tex) & [letter/letter.pdf](file:///Volumes/externalMac/GitHub/Attacks-Delay/letter/letter.pdf) (4 σελίδες, IEEE Communications Letters)**: Ενσωματώθηκε το Θεώρημα 2 (Closed-Form Saddle-Node Fold via Wright-$\omega$) και αφαιρέθηκε το `\resizebox` στον Πίνακα I.
- **Συγχρονισμός Πακέτων arXiv**: Ενημερώθηκαν τα [journal/paper.tex](file:///Volumes/externalMac/GitHub/Attacks-Delay/journal/paper.tex), [arxiv_package/main.tex](file:///Volumes/externalMac/GitHub/Attacks-Delay/arxiv_package/main.tex), `arxiv_submission.tar.gz`, `journal/arxiv_submission.tar.gz` και `letter/arxiv_letter_submission.tar.gz` (`0` Overfull boxes, `0` Type-3 fonts).

---

## 4. Track C — Interactive Web Studio (`web/`) & Subtractive Design Laws `L0`–`L10`

- Δημιουργήθηκε το **Adversarial Queueing & Saddle-Node Bifurcation Studio** στα [index.html](file:///Volumes/externalMac/GitHub/Attacks-Delay/web/index.html), [styles.css](file:///Volumes/externalMac/GitHub/Attacks-Delay/web/styles.css) και [app.js](file:///Volumes/externalMac/GitHub/Attacks-Delay/web/app.js):
  - **Law `L1` & `L8`**: Μπάρα 1 σειράς (`52px`), `15 <= 24` ορατά στοιχεία ελέγχου στην πρώτη απεικόνιση, `3 <= 4` ομάδες επιλογών, `3` μακρο-περιοχές (`S_L8 = 10.0`).
  - **Law `L7` (*Η Απλότητα Ευνοεί την Κανονικότητα*)**: Αυστηρή τυπογραφική και γεωμετρική κανονικότητα (`distinctFontSizes <= 8`, `distinctBorderRadii <= 4`, `distinctGridTemplates <= 5`, `tabular-nums`, $\mathcal{R}_{\text{kanon}} = 1.0000$).
  - **Law `L10` (*Ο Νόμος της Γόνιμης Τριβής*)**: Εισαγωγή πραγματικού προφίλ ριπής πακέτων (`#burst-input` + `#btn-inject-burst`) που προσομοιώνει την τροχιά φάσης (stable basin recovery vs. separatrix escape) και δυναμικός επιλυτής **Two-Station Confidence Cascade** (`solveCascadeDelayJS`).
