// Copyright (2026) Christophe Pallier <christophe@pallier.org>
// Licensed under the Apache License, Version 2.0 (see LICENSE.txt).

// Change Blindness v2.6 — flicker paradigm with 3 changes at once, on
// photographs that come with free-viewing eye tracking (COCO-FreeView), to
// build human "change saliency" maps and compare them with real fixations and
// with the saliency maps of image classifiers.
//
// Stimuli (stimuli/, built by tools/*.py from COCO): each scene is one
// photograph plus several variants that each differ from it by one object,
// recoloured or erased (LaMa inpainting); stimuli/manifest.csv lists them.
// Each variant is stored as the patch where it differs from the original
// (vK_patch.jpg, see tools/make_patches.py). For each trial, 3 variants of the
// scene are drawn at random and their patches pasted onto the original, each
// within its region (vK_region.png), so 3 objects change at once, of either
// type.
//
// The participant clicks on every change they find; each one found is
// framed in green. The trial ends when the 3 are found or after 15 s. Which
// change is found first, second, third (or not at all) ranks the objects of
// the photograph for attention.
//
// Each participant sees 25 scenes drawn at random, in random order, the
// scenes offering more changes being drawn more often (see sceneWeight), and
// for each a random set of 3 changes: different participants get different
// scenes and sets. Whether the original or the
// changed version comes first in the cycle is random. Two scenes that offer
// the fewest changes, the same for everyone, are kept for practice, with
// feedback. Every draw is seeded by the subject ID, so a session can be
// reproduced from its ID.
//
// Trial: fixation cross, then the Rensink flicker loop
//
//	image A 240 ms — blank 80 ms — image A' 240 ms — blank 80 ms — ...
//
// Usage:
//
//	go run . -w -s 1
//
// It also runs in a browser (WebAssembly, see web/ and the README).
package main

import (
	"embed"
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"image"
	"image/draw"
	_ "image/jpeg" // decoders for the stimuli
	_ "image/png"
	"math"
	"math/rand/v2"
	"path"
	"runtime"
	"runtime/debug"
	"sort"
	"strings"
	"time"

	"github.com/chrplr/goxpyriment/apparatus"
	"github.com/chrplr/goxpyriment/control"
	"github.com/chrplr/goxpyriment/stimuli"
)

//go:embed stimuli
var assets embed.FS

const (
	expName = "Change-Blindness-v2.6" // window title and data file names

	// contactEmail is where browser participants send their results.
	contactEmail = "justin.verastegui-munoz@polytechnique.edu"
	stimDir      = "stimuli"

	imageW = 1024 // every stimulus image and mask is 1024×768
	imageH = 768

	nPractice = 2
	nChanges  = 3 // objects that change in each trial

	imageMs    = 240
	blankMs    = 80
	fixationMs = 800
	itiMs      = 700

	// designSeed fixes what must be the same for every participant: the
	// practice scenes.
	designSeed = 0x5a11e4c3
)

var (
	nTrials   = flag.Int("n", 25, "Number of scenes per participant")
	maxMs     = flag.Int("max", 15_000, "Maximum duration of a trial (ms)")
	tolerance = flag.Float64("tolerance", 25, "A click within this distance of a changed object (image pixels) finds it")
	practice  = flag.Bool("practice", true, "Run the practice trials first")

	// The browser page asks for the participant's name; it is recorded in the
	// session's -info.txt only, never in the .csv (which can be shared), and
	// the subject ID that seeds the draw is a random number the page picks.
	lastName  = flag.String("nom", "", "Participant's last name (recorded in the -info.txt only)")
	firstName = flag.String("prenom", "", "Participant's first name (recorded in the -info.txt only)")
)

// variant is one row of stimuli/manifest.csv: a version of a scene that
// differs from the original by one object.
type variant struct {
	scene, id string // e.g. "s139", "v3"
	object    string // COCO category of the changed object
	main      bool   // the changed object is the scene's main object
	change    string // "colour" or "disappear"
	px, py    int    // position of the patch in the photograph
}

func (v variant) patch() string  { return path.Join(stimDir, v.scene, v.id+"_patch.jpg") }
func (v variant) mask() string   { return path.Join(stimDir, v.scene, v.id+"_mask.png") }
func (v variant) region() string { return path.Join(stimDir, v.scene, v.id+"_region.png") }
func original(scene string) string {
	return path.Join(stimDir, scene, "original.jpg")
}

// trial is one scene to present with its changes.
type trial struct {
	phase   string // "practice" or "main"
	scene   string
	changes []variant // nChanges variants, sorted by ID
	order   string    // "original_first" or "changed_first"
}

// readManifest returns the variants of every scene, scenes sorted by ID.
func readManifest() (scenes []string, variants map[string][]variant, err error) {
	f, err := assets.Open(path.Join(stimDir, "manifest.csv"))
	if err != nil {
		return nil, nil, err
	}
	defer f.Close()
	recs, err := csv.NewReader(f).ReadAll()
	if err != nil {
		return nil, nil, fmt.Errorf("reading manifest: %w", err)
	}
	if len(recs) < 2 {
		return nil, nil, fmt.Errorf("manifest is empty")
	}
	col := map[string]int{}
	for i, name := range recs[0] {
		col[name] = i
	}
	for _, name := range []string{"scene", "variant", "object", "main", "change", "patch_x", "patch_y"} {
		if _, ok := col[name]; !ok {
			return nil, nil, fmt.Errorf("manifest has no %q column", name)
		}
	}
	variants = map[string][]variant{}
	for _, r := range recs[1:] {
		v := variant{
			scene: r[col["scene"]], id: r[col["variant"]], object: r[col["object"]],
			main: r[col["main"]] == "1", change: r[col["change"]],
		}
		if _, err := fmt.Sscan(r[col["patch_x"]], &v.px); err != nil {
			return nil, nil, fmt.Errorf("manifest row %s/%s: patch_x: %w", v.scene, v.id, err)
		}
		if _, err := fmt.Sscan(r[col["patch_y"]], &v.py); err != nil {
			return nil, nil, fmt.Errorf("manifest row %s/%s: patch_y: %w", v.scene, v.id, err)
		}
		for _, p := range []string{v.patch(), v.mask(), v.region(), original(v.scene)} {
			if _, err := assets.Open(p); err != nil {
				return nil, nil, fmt.Errorf("manifest row %s/%s: missing %s", v.scene, v.id, p)
			}
		}
		if variants[v.scene] == nil {
			scenes = append(scenes, v.scene)
		}
		variants[v.scene] = append(variants[v.scene], v)
	}
	sort.Strings(scenes)
	for _, s := range scenes {
		if len(variants[s]) < nChanges {
			return nil, nil, fmt.Errorf("scene %s has %d variants, needs %d", s, len(variants[s]), nChanges)
		}
	}
	return scenes, variants, nil
}

// sceneWeight is how much more likely a scene is to be drawn: the number of
// changes it offers beyond the 3 shown (plus one), so a scene offering 3
// changes weighs 1, 5 changes 3, 8 changes 6. Every scene can be drawn, the
// scenes offering more changes (more objects measured, more sets of 3) more
// often.
func sceneWeight(nVariants int) float64 { return float64(nVariants - nChanges + 1) }

// buildTrials returns the practice trials then n main trials for subject.
//
// The nPractice scenes offering the fewest changes (ties in a fixed random
// order) are the practice, the same for everyone. The n main scenes are drawn
// at random for each subject among the others, without replacement, each with
// a probability proportional to sceneWeight (Efraimidis & Spirakis, 2006),
// and shown in random order with a random set of 3 changes in each.
func buildTrials(scenes []string, variants map[string][]variant, n, subject int) ([]trial, []trial, error) {
	design := rand.New(rand.NewPCG(designSeed, 0))
	pool := append([]string(nil), scenes...)
	design.Shuffle(len(pool), func(i, j int) { pool[i], pool[j] = pool[j], pool[i] })
	sort.SliceStable(pool, func(i, j int) bool { return len(variants[pool[i]]) > len(variants[pool[j]]) })
	if len(pool) < nPractice+n {
		return nil, nil, fmt.Errorf("asked for %d scenes plus %d practice, only %d in stimuli/ (use -n)",
			n, nPractice, len(pool))
	}

	rng := rand.New(rand.NewPCG(uint64(subject), designSeed))
	mk := func(phase, s string) trial {
		vs := variants[s]
		perm := rng.Perm(len(vs))[:nChanges]
		sort.Ints(perm)
		t := trial{phase: phase, scene: s}
		for _, k := range perm {
			t.changes = append(t.changes, vs[k])
		}
		t.order = "original_first"
		if rng.IntN(2) == 1 {
			t.order = "changed_first"
		}
		return t
	}
	var prac, main []trial
	for _, s := range pool[len(pool)-nPractice:] {
		prac = append(prac, mk("practice", s))
	}
	// Weighted draw without replacement: the n largest keys u^(1/w).
	type keyed struct {
		scene string
		key   float64
	}
	var cands []keyed
	for _, s := range pool[:len(pool)-nPractice] {
		cands = append(cands, keyed{s, math.Pow(rng.Float64(), 1/sceneWeight(len(variants[s])))})
	}
	sort.SliceStable(cands, func(i, j int) bool { return cands[i].key > cands[j].key })
	chosen := make([]string, n)
	for i := range chosen {
		chosen[i] = cands[i].scene
	}
	rng.Shuffle(len(chosen), func(i, j int) { chosen[i], chosen[j] = chosen[j], chosen[i] })
	for _, s := range chosen {
		main = append(main, mk("main", s))
	}
	return prac, main, nil
}

// mask is a changed object's region: dark pixels of the mask image.
type mask struct {
	in     []bool // imageW*imageH, row-major from the top-left
	area   int
	cx, cy float64 // centroid
	x0, y0 int     // bounding box
	x1, y1 int
}

func decodeAny(p string) (image.Image, error) {
	f, err := assets.Open(p)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		return nil, fmt.Errorf("decoding %s: %w", p, err)
	}
	return img, nil
}

// decode decodes a full-size (1024×768) image.
func decode(p string) (image.Image, error) {
	img, err := decodeAny(p)
	if err != nil {
		return nil, err
	}
	if b := img.Bounds(); b.Dx() != imageW || b.Dy() != imageH {
		return nil, fmt.Errorf("%s is %dx%d, expected %dx%d", p, b.Dx(), b.Dy(), imageW, imageH)
	}
	return img, nil
}

// dark returns, for a full-size mask image (black on white), whether each
// pixel is dark, row-major. Grayscale PNGs are read directly (fast enough for
// the browser build); other formats go through the generic path.
func dark(p string) ([]bool, error) {
	img, err := decode(p)
	if err != nil {
		return nil, err
	}
	out := make([]bool, imageW*imageH)
	if g, ok := img.(*image.Gray); ok {
		for y := 0; y < imageH; y++ {
			row := g.Pix[y*g.Stride : y*g.Stride+imageW]
			for x, v := range row {
				out[y*imageW+x] = v < 128
			}
		}
		return out, nil
	}
	for y := 0; y < imageH; y++ {
		for x := 0; x < imageW; x++ {
			out[y*imageW+x] = gray(img, x, y) < 128
		}
	}
	return out, nil
}

func loadMask(p string) (*mask, error) {
	in, err := dark(p)
	if err != nil {
		return nil, err
	}
	m := &mask{in: in, x0: imageW, y0: imageH, x1: -1, y1: -1}
	var sx, sy float64
	for y := 0; y < imageH; y++ {
		for x := 0; x < imageW; x++ {
			if !in[y*imageW+x] { // white background
				continue
			}
			m.area++
			sx += float64(x)
			sy += float64(y)
			m.x0, m.y0 = min(m.x0, x), min(m.y0, y)
			m.x1, m.y1 = max(m.x1, x), max(m.y1, y)
		}
	}
	if m.area == 0 {
		return nil, fmt.Errorf("%s: empty mask", p)
	}
	m.cx, m.cy = sx/float64(m.area), sy/float64(m.area)
	return m, nil
}

func gray(img image.Image, x, y int) uint32 {
	r, g, b, _ := img.At(x, y).RGBA()
	return (r + g + b) / 3 >> 8
}

// distance returns the distance in pixels from (x, y) to the nearest object
// pixel (0 when the click is on the object).
func (m *mask) distance(x, y float64) float64 {
	ix, iy := int(math.Floor(x)), int(math.Floor(y))
	if ix >= 0 && ix < imageW && iy >= 0 && iy < imageH && m.in[iy*imageW+ix] {
		return 0
	}
	best := math.Inf(1)
	for py := m.y0; py <= m.y1; py++ {
		for px := m.x0; px <= m.x1; px++ {
			if m.in[py*imageW+px] {
				best = min(best, math.Hypot(float64(px)+0.5-x, float64(py)+0.5-y))
			}
		}
	}
	return best
}

// composeRGBA returns the original and the version with every change of t:
// each variant's patch is pasted onto the original within its region.
func composeRGBA(t trial) (orig, changed *image.RGBA, err error) {
	src, err := decode(original(t.scene))
	if err != nil {
		return nil, nil, err
	}
	orig = image.NewRGBA(image.Rect(0, 0, imageW, imageH))
	draw.Draw(orig, orig.Bounds(), src, image.Point{}, draw.Src)
	changed = image.NewRGBA(orig.Bounds())
	copy(changed.Pix, orig.Pix)
	for _, v := range t.changes {
		pimg, err := decodeAny(v.patch())
		if err != nil {
			return nil, nil, err
		}
		patch := image.NewRGBA(image.Rect(0, 0, pimg.Bounds().Dx(), pimg.Bounds().Dy()))
		draw.Draw(patch, patch.Bounds(), pimg, pimg.Bounds().Min, draw.Src)
		reg, err := dark(v.region())
		if err != nil {
			return nil, nil, err
		}
		pw, ph := patch.Bounds().Dx(), patch.Bounds().Dy()
		if v.px < 0 || v.py < 0 || v.px+pw > imageW || v.py+ph > imageH {
			return nil, nil, fmt.Errorf("%s: patch outside the photograph", v.patch())
		}
		for y := 0; y < ph; y++ {
			for x := 0; x < pw; x++ {
				if !reg[(v.py+y)*imageW+v.px+x] {
					continue
				}
				o, q := changed.PixOffset(v.px+x, v.py+y), patch.PixOffset(x, y)
				copy(changed.Pix[o:o+4], patch.Pix[q:q+4])
			}
		}
	}
	return orig, changed, nil
}

// bmp encodes an RGBA image as an uncompressed 24-bit BMP: nothing to
// compress (which matters in the browser), and both images go through the
// same encoder, so outside the changed regions they are identical.
func bmp(img *image.RGBA) []byte {
	w, h := img.Bounds().Dx(), img.Bounds().Dy()
	rowSize := (3*w + 3) &^ 3
	size := 54 + rowSize*h
	buf := make([]byte, size)
	copy(buf[0:2], "BM")
	binary.LittleEndian.PutUint32(buf[2:], uint32(size))
	binary.LittleEndian.PutUint32(buf[10:], 54)
	binary.LittleEndian.PutUint32(buf[14:], 40)
	binary.LittleEndian.PutUint32(buf[18:], uint32(w))
	binary.LittleEndian.PutUint32(buf[22:], uint32(h)) // positive height: rows bottom-up
	binary.LittleEndian.PutUint16(buf[26:], 1)
	binary.LittleEndian.PutUint16(buf[28:], 24)
	binary.LittleEndian.PutUint32(buf[34:], uint32(rowSize*h))
	for y := 0; y < h; y++ {
		row := buf[54+(h-1-y)*rowSize:]
		src := img.Pix[y*img.Stride:]
		for x := 0; x < w; x++ {
			row[3*x], row[3*x+1], row[3*x+2] = src[4*x+2], src[4*x+1], src[4*x]
		}
	}
	return buf
}

// composeImages returns the original and the changed version of t, encoded
// for stimuli.NewPictureFromMemory.
func composeImages(t trial) (orig, changed []byte, err error) {
	a, b, err := composeRGBA(t)
	if err != nil {
		return nil, nil, err
	}
	return bmp(a), bmp(b), nil
}

// found records when and where a change was found.
type found struct {
	ok             bool
	rank           int     // 1 = found first
	rtMs           float64 // from the first image onset
	clickX, clickY float64 // image pixels, origin at the top-left
	dist           float64
}

// result of one flicker trial.
type result struct {
	changes     []found // same order as trial.changes
	falseClicks []string
	durationMs  float64
	cycles      int
}

func framesFor(ms int, frame time.Duration) int {
	return max(1, int(math.Round(float64(time.Duration(ms)*time.Millisecond)/float64(frame))))
}

// frames draws a 4 px green frame around every object found so far.
func frames(s *apparatus.Screen, masks []*mask, res *result) error {
	for k, m := range masks {
		if !res.changes[k].ok {
			continue
		}
		if err := drawFrame(s, m, 0, 200, 0); err != nil {
			return err
		}
	}
	return nil
}

func drawFrame(s *apparatus.Screen, m *mask, r, g, b uint8) error {
	if err := s.Renderer.SetDrawColor(r, g, b, 255); err != nil {
		return err
	}
	for i := float32(0); i < 4; i++ {
		rect := control.FRect{
			X: float32(m.x0) - 12 - i, Y: float32(m.y0) - 12 - i,
			W: float32(m.x1-m.x0) + 24 + 2*i, H: float32(m.y1-m.y0) + 24 + 2*i,
		}
		if err := s.Renderer.RenderRect(&rect); err != nil {
			return err
		}
	}
	return nil
}

// flicker runs the A–blank–A'–blank loop, taking clicks, until every change
// is found or the time is up.
func flicker(exp *control.Experiment, a, b *stimuli.Picture, masks []*mask) (result, error) {
	frame := exp.Screen.FrameDuration()
	nImg, nBlank := framesFor(imageMs, frame), framesFor(blankMs, frame)
	cycle := 2 * (nImg + nBlank)
	maxFrames := framesFor(*maxMs, frame)
	res := result{changes: make([]found, len(masks))}

	gc := debug.SetGCPercent(-1)
	defer debug.SetGCPercent(gc)

	exp.PollEvents(nil) // drop clicks made before the onset
	var onset, last uint64
	nFound := 0
	for f := 0; f < maxFrames; f++ {
		if err := exp.Screen.Clear(); err != nil {
			return res, err
		}
		var pic *stimuli.Picture
		switch k := f % cycle; {
		case k < nImg:
			pic = a
		case k >= nImg+nBlank && k < 2*nImg+nBlank:
			pic = b
		}
		if pic != nil {
			if err := pic.Draw(exp.Screen); err != nil {
				return res, err
			}
			if err := frames(exp.Screen, masks, &res); err != nil {
				return res, err
			}
		}
		ts, err := exp.Screen.FlipTS()
		if err != nil {
			return res, err
		}
		if f == 0 {
			onset = ts
		}
		last = ts
		res.cycles = f / cycle

		state := exp.PollEvents(nil)
		if state.QuitRequested {
			return res, control.EndLoop
		}
		if state.LastMouseButton != control.BUTTON_LEFT {
			continue
		}
		mx, my := exp.Screen.MousePosition()
		x, y := float64(mx)+imageW/2, imageH/2-float64(my)
		rt := float64(int64(state.LastMouseTimestamp)-int64(onset)) / 1e6
		best, bestD := -1, math.Inf(1)
		for k, m := range masks {
			if res.changes[k].ok {
				continue
			}
			if d := m.distance(x, y); d < bestD {
				best, bestD = k, d
			}
		}
		if best < 0 || bestD > *tolerance {
			res.falseClicks = append(res.falseClicks, fmt.Sprintf("%.0f:%.0f:%.0f", x, y, rt))
			continue
		}
		nFound++
		res.changes[best] = found{ok: true, rank: nFound, rtMs: rt, clickX: x, clickY: y, dist: bestD}
		if nFound == len(masks) {
			break
		}
	}
	res.durationMs = float64(int64(last)-int64(onset)) / 1e6
	return res, nil
}

// showFeedback shows the original with the changes framed: green when found,
// red when missed (practice only).
func showFeedback(exp *control.Experiment, b *stimuli.Picture, masks []*mask, res result) error {
	n := 0
	for _, c := range res.changes {
		if c.ok {
			n++
		}
	}
	msg := fmt.Sprintf("Vous avez trouvé %d changement(s) sur %d. En rouge, ceux qui manquaient.", n, len(masks))
	if n == len(masks) {
		msg = "Bravo, vous les avez tous trouvés !"
	}
	label := stimuli.NewTextLine(msg, 0, imageH/2-40, control.Yellow)
	defer label.Unload()
	// Logical size = image size, so image pixels are SDL logical pixels.
	scene := stimuli.DrawFunc(func(s *apparatus.Screen) error {
		if err := b.Draw(s); err != nil {
			return err
		}
		for k, m := range masks {
			r, g := uint8(0), uint8(200)
			if !res.changes[k].ok {
				r, g = 255, 0
			}
			if err := drawFrame(s, m, r, g, 0); err != nil {
				return err
			}
		}
		return label.Draw(s)
	})
	if err := exp.Show(scene); err != nil {
		return err
	}
	return exp.Wait(3000)
}

func main() {
	exp := control.NewExperimentFromFlags(expName, control.Gray, control.White, 28)
	defer exp.End()
	if *lastName != "" || *firstName != "" {
		exp.AddExperimentInfo("participant_nom: " + strings.TrimSpace(*lastName))
		exp.AddExperimentInfo("participant_prenom: " + strings.TrimSpace(*firstName))
	}

	if err := exp.SetLogicalSize(imageW, imageH); err != nil {
		exp.Fatal("set logical size: %v", err)
	}
	if err := exp.ShowCursor(); err != nil {
		exp.Fatal("show cursor: %v", err)
	}

	scenes, variants, err := readManifest()
	if err != nil {
		exp.Fatal("%v", err)
	}
	prac, trials, err := buildTrials(scenes, variants, *nTrials, exp.SubjectID)
	if err != nil {
		exp.Fatal("%v", err)
	}
	if *practice {
		trials = append(prac, trials...)
	}

	exp.AddDataVariableNames([]string{
		"phase", "trial", "scene", "changes", "order",
		"variant", "object", "main_object", "change",
		"found", "rank", "rt_ms", "click_x", "click_y", "dist_px",
		"mask_cx", "mask_cy", "mask_area",
		"n_found", "false_clicks", "trial_ms", "cycles",
	})

	fix := stimuli.NewFixCross(30, 3, control.White)

	err = exp.Run(func() error {
		if err := exp.ShowInstructions(
			"DÉTECTION DE CHANGEMENTS\n\n" +
				"Une photo va clignoter entre deux versions.\n" +
				"Entre les deux, TROIS objets changent :\n" +
				"ils apparaissent, disparaissent ou changent de couleur.\n\n" +
				"CLIQUEZ DIRECTEMENT SUR CHAQUE CHANGEMENT\n" +
				"dès que vous le repérez. Un cadre vert confirme\n" +
				"chaque changement trouvé.\n\n" +
				fmt.Sprintf("Vous avez %d secondes par photo.\n", *maxMs/1000) +
				"Soyez rapide, mais ne cliquez pas au hasard.\n\n" +
				"Appuyez sur ESPACE pour commencer."); err != nil {
			return err
		}

		trialNum := 0
		for i, t := range trials {
			if t.phase == "main" && i > 0 && trials[i-1].phase == "practice" { // only after a practice
				if err := exp.ShowInstructions(fmt.Sprintf(
					"Fin de l'entraînement.\n\nL'expérience comporte %d photos.\n\n"+
						"Appuyez sur ESPACE pour commencer.", *nTrials)); err != nil {
					return err
				}
				trialNum = 0
			}
			trialNum++

			origPNG, changedPNG, err := composeImages(t)
			if err != nil {
				return err
			}
			orig := stimuli.NewPictureFromMemory(origPNG, 0, 0)
			changed := stimuli.NewPictureFromMemory(changedPNG, 0, 0)
			a, b := orig, changed
			if t.order == "changed_first" {
				a, b = changed, orig
			}
			masks := make([]*mask, len(t.changes))
			for k, v := range t.changes {
				if masks[k], err = loadMask(v.mask()); err != nil {
					return err
				}
			}
			// Upload both textures now, so the first flicker frames do not
			// pay for decoding.
			if err := stimuli.PreloadVisualOnScreen(exp.Screen, a); err != nil {
				return err
			}
			if err := stimuli.PreloadVisualOnScreen(exp.Screen, b); err != nil {
				return err
			}

			if err := exp.Show(fix); err != nil {
				return err
			}
			if err := exp.Wait(fixationMs); err != nil {
				return err
			}

			res, err := flicker(exp, a, b, masks)
			if err != nil {
				return err
			}
			if t.phase == "practice" {
				if err := showFeedback(exp, orig, masks, res); err != nil {
					return err
				}
			}
			orig.Unload()
			changed.Unload()

			nFound := 0
			ids := make([]string, len(t.changes))
			for k, v := range t.changes {
				ids[k] = v.id
				if res.changes[k].ok {
					nFound++
				}
			}
			for k, v := range t.changes {
				c, m := res.changes[k], masks[k]
				rt, cx, cy, dist := float64(*maxMs), -1.0, -1.0, -1.0
				if c.ok {
					rt, cx, cy, dist = c.rtMs, c.clickX, c.clickY, c.dist
				}
				exp.Data.Add(t.phase, trialNum, t.scene, strings.Join(ids, "|"), t.order,
					v.id, v.object, v.main, v.change,
					c.ok, c.rank, fmt.Sprintf("%.1f", rt),
					fmt.Sprintf("%.1f", cx), fmt.Sprintf("%.1f", cy), fmt.Sprintf("%.1f", dist),
					fmt.Sprintf("%.1f", m.cx), fmt.Sprintf("%.1f", m.cy), m.area,
					nFound, strings.Join(res.falseClicks, "|"), fmt.Sprintf("%.1f", res.durationMs), res.cycles)
			}
			var report []string
			for k, v := range t.changes {
				s := fmt.Sprintf("%s %s/%s: -", v.id, v.object, v.change)
				if c := res.changes[k]; c.ok {
					s = fmt.Sprintf("%s %s/%s: #%d %.1f s", v.id, v.object, v.change, c.rank, c.rtMs/1000)
				}
				report = append(report, s)
			}
			fmt.Printf("%-8s %2d  %-8s %d/%d found, %d false  [%s]\n", t.phase, trialNum, t.scene,
				nFound, len(t.changes), len(res.falseClicks), strings.Join(report, "; "))

			if err := exp.Blank(itiMs); err != nil {
				return err
			}
		}

		end := "Expérience terminée.\nMerci pour votre participation !\n\nAppuyez sur ESPACE pour quitter."
		if runtime.GOOS == "js" { // browser: the results are downloaded when the session ends
			end = "Expérience terminée. Merci pour votre participation !\n\n" +
				"Appuyez sur ESPACE : votre navigateur va télécharger\n" +
				"un fichier .zip qui contient vos résultats.\n\n" +
				"Envoyez ce fichier par e-mail à\n" + contactEmail + "\n\n" +
				"(il est en général dans votre dossier Téléchargements)."
		}
		if err := exp.ShowInstructions(end); err != nil {
			return err
		}
		return control.EndLoop
	})
	if err != nil && !control.IsEndLoop(err) {
		exp.Fatal("experiment error: %v", err)
	}
}
