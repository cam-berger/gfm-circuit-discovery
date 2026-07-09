# Experiment log, in plain language

What we did, what it showed, and why it mattered — written so it can be read without knowing the
jargon. Each entry links to the formal record in `DECISIONS.md` and the code that reproduces it.

---

## E1 — Two fake models with known wiring (Rung 0)

**What we did.** Before pointing a new tool at a real DNA model, we built two toy models *by hand*,
setting every internal connection ourselves, so we knew exactly how each one worked. Both do the
same job — figure out where a gene's three-letter reading groups begin — and both get it right
equally often. But one does it honestly (find the start marker, count from there) and the other
cheats (memorise which letters tend to sit in which position).

**Why.** A new tool that draws you a picture of a model's inner workings is useless until you know
the picture is right. You calibrate a thermometer in ice water first. And a tool that draws a
*convincing* picture of every model is worse than useless, so we needed a model where the honest
answer is "there's no real mechanism here" — that's the cheating twin.

**What it showed.** The two models behave identically (accuracy within 0.003 of each other), and
delete their shared start marker and the honest one collapses from 100% to 38% while the cheat is
completely unaffected. So they really are the same on the outside and different on the inside.

*Formal record: D2, D3. Code: `src/gfmcd/models/planted.py`, `tests/test_planted.py`.*

---

## E2 — The tool is exact, and it is blind (Test A)

**What we did.** Pointed the tracing tool at the honest twin and checked whether the map it drew
matched the blueprint we had written.

**What it showed.** Where the tool works, it is perfect — the numbers add up to about twenty decimal
places, and it found every connection we planted, inventing none.

But it missed the single most important one. Our model does two things: it **finds** the start
marker, then it **counts** from it. The tool drew the counting beautifully and gave the finding a
score of exactly zero — "this doesn't matter." Deleting that marker drops accuracy from 100% to 38%,
so it matters enormously.

The reason is structural: to do its arithmetic, the tool has to freeze the model's attention — where
it is looking — in place. That's like studying how someone reads by freezing their eyes. You capture
what they do with the words they're staring at and miss entirely how they chose where to look. Every
question we care about in DNA — find the gene start, find the splice site — involves choosing where
to look.

*Formal record: D12, D13. Code: `src/gfmcd/tracer/graph.py`, `tests/test_a_attribution.py`.*

---

## E3 — The obvious fix fails silently (Test A2)

**What we did.** Built a second instrument to see the connections the first one was blind to.

**What it showed.** The obvious approach returns **zero for everything** — and not because of a bug.
Our model's marker-finding is *confident*: it is completely certain where to look. The standard
mathematical trick for measuring importance works by nudging things slightly and watching what
changes, and nudging a completely certain decision changes nothing. **The tool reports the connection
as irrelevant precisely because it is so relevant that the model is sure.**

This would have been invisible on a real DNA model. We'd have got an empty picture and written it up
as an honest negative result. The fix is to measure along a path from "no information at all" up to
the real model, instead of only at the confident endpoint. With that, the tool recovers the missing
connection, puts 100% of the credit on the right letter, and correctly reports *nothing* for a
control model where the same machinery exists but nothing downstream uses it.

*Formal record: D20. Code: `src/gfmcd/tracer/qk.py`, `tests/test_qk_attribution.py`.*

---

## E4 — Perfect reconstruction, meaningless parts (Test C)

**What we did.** Models don't store ideas in neat labelled boxes; they cram many concepts into
overlapping space. We used the standard method for untangling them, on a model whose three concepts
we had planted ourselves.

**What it showed.** The method rebuilt the model's behaviour *perfectly* and recovered none of its
concepts. One of the three concepts vanished entirely — it had been absorbed into a background
constant, so it simply had no part of its own. Another "part" turned out not to be a concept at all
but a *difference* between two of them.

If this had happened on a real model, we'd have looked at what that part responds to, seen a sensible
biological pattern, and given it a confident name — while a whole concept sat invisible in the
background. The rule we now follow: trust the **shape** of the diagram, don't trust the **names** of
its parts without a separate check.

*Formal record: D14, D17. Code: `tests/test_c_end_to_end.py`.*

---

## E5 — Reading real DNA, and the trap we avoided (`data/genome.py`)

**What we did.** DNA is read three letters at a time, like `THE CAT ATE` — so every letter is either
the 1st, 2nd, or 3rd of its group. `genome.py` cuts real human DNA into chunks and labels each letter
with its position in the group.

**Are the labels right?** We checked against biology rather than against our own code. Three specific
three-letter combinations mean "stop reading here," and they should essentially never appear inside a
gene when you read in the correct grouping — but they turn up constantly if you're off by one letter.
Reading 117,481 groups using our labels gave **zero** stop signals. Shift by one letter and 4–6% are
stops. That is the biology independently confirming the arithmetic, including the genuinely tricky
case where a gene is read backwards along the opposite strand of the DNA.

```
in-frame (our labels)   0.0000 stop codons   n = 117,481
shifted by one letter   0.0419
shifted by two letters  0.0554
```

**The trap.** If we had cut every chunk starting exactly at a gene's beginning, then "1st letter of
the group" would always land on positions 1, 4, 7, 10… The model would never need to understand
anything — it could just count. It would *look* like it had learned to read DNA. Its training numbers
would look perfect. It would have learned nothing, and no loss curve would ever reveal it.

We tested this by deliberately writing the broken version. Cutting at gene starts lets you predict the
answer from position alone **100% of the time**. Cutting at random starting points drops that to
**34.5%** — essentially chance, since there are three possibilities.

**Why it matters.** When our model now works out the reading frame, we will know it did so by finding
the gene's starting point and counting from there — the thing we actually want to trace — and not by
exploiting an accident in how we prepared the data.

*Formal record: D21. Code: `src/gfmcd/data/genome.py`, `scripts/validate_genome.py`.*

---

## E6 — The model takes the shortcut, because the shortcut is good enough

**What we did.** Trained a model on real human DNA and asked whether it figures out the reading frame
by finding the gene's start marker and counting, or by guessing from the local flavour of the letters.

**How we tested it.** Deleting the start marker turned out to prove nothing — the model just falls back
on the local flavour, like a reader who loses their place and re-reads the sentence. So instead we made
the two clues *disagree*: insert one extra letter right after the start marker, so that counting from
the marker says one thing and the local flavour says another. The model has to pick a side.

**What it showed.** Beyond about 30 letters, the model sides with the local flavour, every time. And a
control model that had *never seen a start marker in training* behaves identically — proving the
apparent marker-following near the insertion is just the model pooling letters from both sides of it.

**Why.** We measured how much the training goal actually rewards knowing the frame: about 0.4% of the
model's total error. The shortcut is nearly as good as the real thing. **The model isn't failing — it
is correctly solving the problem we gave it.** Training it longer would not help; there is nothing to
find.

*Formal record: D24. Code: `scripts/frameshift_conflict.py`.*

---

## E7 — Make the circuit necessary, and it appears all at once

**What we did.** Built an artificial version of the problem where the shortcut is impossible. The DNA
is completely random, so the local flavour tells you nothing at all about the frame. But we ask the
model to name the protein letter at every position — which you simply cannot do unless you know where
the reading groups start. The only way is to find the start marker and count.

**What it showed.** For 2,750 rounds of training the model was stuck at exactly the score you get by
guessing (0.377). Then, over about 500 rounds, it jumped to 0.999.

That sudden jump is the model discovering the circuit. There is no halfway: you cannot half-know where
the reading groups begin.

**And now every test we ran on real DNA gives the opposite answer.** Delete the start marker and the
model collapses to guessing. Delete an identical-looking marker elsewhere and nothing happens. Make the
clues disagree and the model sides with the start marker at *every* distance, even 250 letters away —
it propagates the shift exactly as real biology does. The frame information also gets *built up* through
the model's layers rather than appearing instantly, which is what a multi-step computation looks like.

**The point.** A model builds the real machinery exactly when the task makes it necessary — not when it
is merely useful. Real genomic training sits on the wrong side of that line.

*Formal record: D25, D26. Code: `src/gfmcd/data/uniform_cds.py`, `scripts/verify_uniform.py`.*

---

## The thread running through all of these

Every one of these experiments found a problem *before* we touched a real genomic model: a blind spot
in the method, a measurement that silently returns zero, an untangling procedure that names things
wrongly, and a data-preparation trap that would have faked success. Each would have produced results
that looked entirely plausible.

That is the argument for building the ice water first.
