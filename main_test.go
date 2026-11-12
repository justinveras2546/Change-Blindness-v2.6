// Copyright (2026) Christophe Pallier <christophe@pallier.org>
// Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

package main

import (
	"image"
	"testing"
)

func TestManifestAndDesign(t *testing.T) {
	scenes, variants, err := readManifest()
	if err != nil {
		t.Fatal(err)
	}

	n := *nTrials // the default session
	prac1, main1, err := buildTrials(scenes, variants, n, 1)
	if err != nil {
		t.Fatal(err)
	}
	prac2, main2, err := buildTrials(scenes, variants, n, 2)
	if err != nil {
		t.Fatal(err)
	}
	if len(prac1) != nPractice || len(main1) != n {
		t.Fatalf("got %d practice and %d main trials, want %d and %d", len(prac1), len(main1), nPractice, n)
	}
	for i := range prac1 {
		if prac1[i].scene != prac2[i].scene {
			t.Errorf("practice scenes differ between subjects: %s vs %s", prac1[i].scene, prac2[i].scene)
		}
	}

	// Practice: scenes offering the fewest changes. Main scenes: a weighted
	// draw, different between subjects, every scene drawn at some point, the
	// scenes offering more changes more often.
	fewest := len(variants[scenes[0]])
	for _, s := range scenes {
		fewest = min(fewest, len(variants[s]))
	}
	for _, tr := range prac1 {
		if len(variants[tr.scene]) != fewest {
			t.Errorf("practice scene %s offers %d changes, not the fewest (%d)", tr.scene, len(variants[tr.scene]), fewest)
		}
	}
	const subjects = 400
	shownCount := map[string]int{}
	for s := 1; s <= subjects; s++ {
		_, m, _ := buildTrials(scenes, variants, n, s)
		for _, tr := range m {
			shownCount[tr.scene]++
		}
	}
	isPractice := map[string]bool{}
	for _, tr := range prac1 {
		isPractice[tr.scene] = true
	}
	rate := map[int][]float64{} // changes offered -> share of sessions showing the scene
	for _, s := range scenes {
		if isPractice[s] {
			continue
		}
		if shownCount[s] == 0 {
			t.Errorf("scene %s never drawn in %d sessions", s, subjects)
		}
		rate[len(variants[s])] = append(rate[len(variants[s])], float64(shownCount[s])/subjects)
	}
	mean := func(xs []float64) float64 {
		sum := 0.0
		for _, x := range xs {
			sum += x
		}
		return sum / float64(len(xs))
	}
	for k := 3; k <= 8; k++ {
		if len(rate[k]) > 0 {
			t.Logf("scenes offering %d changes (%2d): shown in %3.0f %% of sessions", k, len(rate[k]), 100*mean(rate[k]))
		}
	}
	if len(rate[3]) > 0 && len(rate[5]) > 0 && mean(rate[5]) <= mean(rate[3]) {
		t.Errorf("scenes offering 5 changes are not drawn more often than those offering 3")
	}
	same := 0
	for i := range main1 {
		if main1[i].scene == main2[i].scene {
			same++
		}
	}
	if same == len(main1) {
		t.Error("subjects 1 and 2 got the same scenes in the same order")
	}

	seen := map[string]bool{}
	for _, tr := range append(prac1, main1...) {
		if seen[tr.scene] {
			t.Errorf("scene %s shown twice to subject 1", tr.scene)
		}
		seen[tr.scene] = true
		if len(tr.changes) != nChanges {
			t.Fatalf("scene %s: %d changes, want %d", tr.scene, len(tr.changes), nChanges)
		}
		ids := map[string]bool{}
		for _, v := range tr.changes {
			if v.scene != tr.scene || ids[v.id] {
				t.Errorf("scene %s: bad or repeated change %s/%s", tr.scene, v.scene, v.id)
			}
			ids[v.id] = true
		}
	}

	// Scenes with more than 3 variants should not always give everyone the
	// same set.
	set := func(tr trial) string {
		s := ""
		for _, v := range tr.changes {
			s += v.id
		}
		return s
	}
	shown := map[string]string{}
	for _, tr := range main1 {
		shown[tr.scene] = set(tr)
	}
	differ, shared := 0, 0
	for _, tr := range main2 {
		if s, ok := shown[tr.scene]; ok && len(variants[tr.scene]) > nChanges {
			shared++
			if s != set(tr) {
				differ++
			}
		}
	}
	if shared > 0 && differ == 0 {
		t.Error("subjects 1 and 2 got the same changes in every scene they share")
	}

	again, _, _ := buildTrials(scenes, variants, n, 1)
	for i := range again {
		if again[i].scene != prac1[i].scene || set(again[i]) != set(prac1[i]) {
			t.Fatal("the same subject ID does not give the same session")
		}
	}
}

// The original and the changed version must be identical outside the
// changed regions, or the flicker would show more than the 3 changes.
func TestComposeOnlyChangesRegions(t *testing.T) {
	scenes, variants, err := readManifest()
	if err != nil {
		t.Fatal(err)
	}
	_, main1, err := buildTrials(scenes, variants, 1, 7)
	if err != nil {
		t.Fatal(err)
	}
	tr := main1[0]
	ia, ib, err := composeRGBA(tr)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := composeImages(tr); err != nil {
		t.Fatal(err)
	}
	var regions []image.Image
	for _, v := range tr.changes {
		r, err := decode(v.region())
		if err != nil {
			t.Fatal(err)
		}
		regions = append(regions, r)
	}
	changedInside := 0
	for y := 0; y < imageH; y++ {
		for x := 0; x < imageW; x++ {
			inside := false
			for _, r := range regions {
				inside = inside || gray(r, x, y) < 128
			}
			same := ia.At(x, y) == ib.At(x, y)
			if !inside && !same {
				t.Fatalf("scene %s: pixel (%d,%d) differs outside the changed regions", tr.scene, x, y)
			}
			if inside && !same {
				changedInside++
			}
		}
	}
	if changedInside == 0 {
		t.Errorf("scene %s: nothing changed", tr.scene)
	}
}
