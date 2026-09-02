# Academic Publication & Submission Guide

This document outlines the step-by-step procedure for submitting the manuscript to **arXiv** and **IEEE Transactions on Networking (ToN)**.

---

## 1. Quick arXiv Submission (Immediate Pre-Print)

### Step-by-Step Instructions:
1. Navigate to [arxiv.org/submit](https://arxiv.org/submit) and log in.
2. Select **"Start New Submission"**.
3. **Upload Files**: Drag and drop [`arxiv_submission.tar.gz`](./arxiv_submission.tar.gz).
4. **Metadata Entry** (Copy-paste directly from [`ARXIV_METADATA.txt`](./ARXIV_METADATA.txt)):
   * **Title**: `Mathematical Modeling and Performance Analysis of Multi-Node Queueing Networks Under Adversarial Attacks and Timeout-Driven Retransmissions`
   * **Authors**: `Ilias Chrysovergis`
   * **Primary Category**: `cs.NI` (Networking and Internet Architecture)
   * **Cross-Lists**: `math.PR` (Probability), `cs.CR` (Cryptography and Security)
   * **Abstract**: Paste the text from [`ARXIV_METADATA.txt`](./ARXIV_METADATA.txt).
5. **View PDF & Approve**: Verify the generated proof and submit.

---

## 2. Peer-Reviewed Journal Submission: IEEE Transactions on Networking (ToN)

### Target Venue Details:
* **Journal**: *IEEE/ACM Transactions on Networking (ToN)* (or *IEEE Transactions on Communications (TCOM)*)
* **Submission Portal**: [mc.manuscriptcentral.com/ton-ieee](https://mc.manuscriptcentral.com/ton-ieee)
* **Document Files**:
  * **LaTeX Source**: [`paper_ieee.tex`](./paper_ieee.tex) (compiled with `IEEEtran.cls`)
  * **Manuscript PDF**: [`paper_ieee.pdf`](./paper_ieee.pdf) (6 pages, two-column IEEE format)

### Submission Steps:
1. Log in to ScholarOne Manuscripts for IEEE ToN.
2. Select **"Author"** $\to$ **"Start New Submission"**.
3. Choose manuscript type: **Regular Paper**.
4. Upload [`paper_ieee.pdf`](./paper_ieee.pdf) as the primary PDF document.
5. In the metadata section:
   * **Keywords**: `Queueing theory, adversarial networks, renewal-reward process, fixed-point traffic analysis, hypoexponential distributions, moment-matching, Erlang distribution, network stability`.
   * **Corresponding Author**: Ilias Chrysovergis (`iliachry@gmail.com`).
6. Complete the mandatory IEEE copyright and conflict-of-interest declarations, review PDF proof, and submit.

---

## 3. Artifacts Summary in this Repository

| File | Purpose |
| :--- | :--- |
| [`paper.tex`](./paper.tex) / [`paper.pdf`](./paper.pdf) | Comprehensive 16-page full theoretical manuscript (single-column format) |
| [`paper_ieee.tex`](./paper_ieee.tex) / [`paper_ieee.pdf`](./paper_ieee.pdf) | IEEE Transactions camera-ready 6-page two-column version |
| [`arxiv_submission.tar.gz`](./arxiv_submission.tar.gz) | Complete, standalone drag-and-drop archive for arXiv submission |
| [`ARXIV_METADATA.txt`](./ARXIV_METADATA.txt) | Formatted metadata for quick copy-pasting on arXiv |
| `1_one_node_destruction/` $\dots$ `5_n_node_feedback/` | Python simulation scripts & reproducible `.json` data files |
