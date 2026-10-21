# 24 October 2026 course-submission checklist

This checklist follows `C:\Users\Laptop\Downloads\FISAC QP.odt` (read on 21 October 2026) and the saved repository evidence. The proposal PDF is the visual template for the final report. Any separate final rubric, professor feedback, and exact LMS cutoff have not been supplied in the checkout. Complete the report and slides before the fixed 24 October deadline; the active research-validation workstream is tracked in `future-paper-gap.md` and may continue after upload where evidence is still unavailable.

## Report source and required sections

- [ ] **Use the supplied proposal PDF as the report's visual template.** A final `.tex` project is absent from this checkout. Build it in IEEE style, then compile and visually inspect the final PDF against the supplied proposal and course requirements.
- [ ] **Introduction, at least one page.** Cover IoT, AIoT, application domain, importance, existing challenges, motivation, precise synthetic-study problem, objectives, and a modest contribution statement. Include a properly labelled sensing–gateway–processing–decision figure; distinguish proposed hardware from the implemented offline simulation.
- [ ] **Literature survey, at least one page.** Critically compare at least three relevant papers, including at least one journal article. For each, check the problem, IoT architecture, AI method, dataset, methodology, metrics/results, strengths, weaknesses, and remaining gap. End with a comparison table and a literature-supported research gap. The existing proposal bibliography is not a verified critical survey; the team must supply or verify the selected full texts and references. Do not claim a novel model combination without evidence.
- [ ] **Methodology, approximately two pages.** Explain the complete data flow, all five assigned attack families plus the drift ramp, training-only preprocessing, source-disjoint artificial episodes, fixed batch splits, AE/RF/Isolation Forest decisions, validation selection, and evaluation denominators. Include a readable architecture diagram based on `submission_note.md`, with “implemented offline” and “proposed embedded placement” clearly marked. Explain why compensation/quarantine and device/edge/cloud execution are not implementation results.
- [ ] **Implementation and evaluation.** Show Track A coverage separately from Track B synthetic labels. Use saved confusion matrices and denominator-bearing tables/figures from `reports/final_results/`; keep binary detection and three-class classification in separate panels. Present matched-primary as primary and original stress as a related sensitivity condition. State that Batches 7–10 are exploratory and were not used to select models or thresholds.
- [ ] **Results interpretation and conclusion.** Include the exact RF/AE-gated RF values, 724/909 versus 752/909 background intervention flags, weak bias/scaling/noise classification, and freeze/replay control caveat. State artificial episode order, known onset, gas/condition shift, no real aging or attack labels, and no demonstrated three-class benefit from the AE gate. Summarize achieved and unachieved objectives and give future work without claiming measured improvement or latency.
- [ ] **References and attribution.** Use IEEE reference format; verify each cited paper and figure/table source. Cite UCI data and distinguish the course proposal from tested evidence. Follow the institution's academic-integrity and AI-use rules.
- [ ] **Final similarity report.** Obtain a Turnitin report for the actual final PDF/source according to institutional rules. An earlier similarity PDF recorded in provenance would not establish similarity for a newly compiled final submission.

## Five-minute presentation per member

- [ ] Allocate each member a five-minute contribution consistent with the course outline: context/problem; literature and gap; architecture/method; implementation/evaluation; results/conclusion. Rebalance the supplied outline across the actual team.
- [ ] Use one architecture slide that visibly separates implemented offline steps from proposed sensing, gateway, edge, and cloud blocks.
- [ ] Use one compact results slide with the four-row task-separated table and five family bars in `submission_note.md`. Keep denominators and the “exploratory synthetic test” label visible. Do not compare binary and three-class macro-F1 as the same task.
- [ ] Rehearse timing and questions on false alarms, exact-copy controls, unseen toluene in training, missing real labels, and the AE gate's negative result.

## Package and verification before upload

- [x] Read-only verification path and temporary generation tests are in place. The 34 frozen `reports/` files match the original committed contents under the documented LF/CRLF policy.
- [x] Run `scripts.verify_frozen` with Python 3.10+ and record its exit code. It now exits `0` locally: all four external inputs and 34 frozen reports pass. Rerun it before upload; this does not establish fresh model reproduction.
- [ ] Cross-check the final PDF, slides, captions, class order, numbers, and citations against the frozen reports. Verify figure legibility after compilation and confirm that no saved result or preflight file changed.
- [ ] Confirm the exact LMS cutoff, upload format, team roster/professor approval, and any separate final rubric. Upload the compiled report, required source/figures, slides, and final similarity report as directed by the course.

## Files and decisions the team must supply or verify

1. Prescribed LaTeX template, final report source/PDF, slide deck, and final-document Turnitin report; none is in this checkout.
2. Selected paper full texts or verified citation details, plus a defensible literature comparison and gap. The proposal's references alone do not complete this requirement.
3. Exact LMS deadline and upload rules, professor feedback, team/approval status, and any additional course rubric.
4. For *fresh reproduction*: the pinned `data/raw/uci-270.zip` is present locally, and an ignored Windows Python 3.13.16 environment passes the exact package gate. The three root-level provenance PDFs are present and required for strict historical verification, but are not model inputs. The isolated Windows run stopped at a preflight numeric difference and also produced different RF selection and matrices. The team must disclose that fresh episode predictions and gas-specific error rates are unverified; preserve the frozen course results. The archive, outer ZIP, PDFs, and ignored environment need deliberate review before any inclusion in a code submission.
