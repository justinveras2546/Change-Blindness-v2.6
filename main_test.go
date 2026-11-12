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

	const n = 40
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
