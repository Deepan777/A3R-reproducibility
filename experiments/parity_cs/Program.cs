// E8b - Run the ORIGINAL A3R C# feature extractor and adaptation controller
// (copied unmodified from the repository) on the exported inputs; write
// outputs for comparison with the Python reference and time both routines.
// Usage: dotnet run -c Release -- <parity_dir>
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Reflection;
using A3R.Adaptation;
using A3R.Affect;
using A3R.ARGuidance;

static class Program
{
    static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

    static int Main(string[] args)
    {
        string dir = args[0];
        Features(dir);
        Controller(dir);
        return 0;
    }

    static void Features(string dir)
    {
        byte[] raw = File.ReadAllBytes(Path.Combine(dir, "landmarks_f32.bin"));
        int per = 478 * 3, n = raw.Length / 4 / per;
        var sets = new float[n][];
        for (int i = 0; i < n; i++) { sets[i] = new float[per]; Buffer.BlockCopy(raw, i * per * 4, sets[i], 0, per * 4); }
        using (var w = new StreamWriter(Path.Combine(dir, "cs_features.csv")))
        {
            w.WriteLine("i,B,E,M,yaw,pitch,gated,C,F");
            for (int i = 0; i < n; i++)
            {
                var f = AffectFeatureExtractor.Extract(sets[i]);
                // score formulas exactly as in AffectScorer.ProcessFrame (single precision)
                float C = UnityEngine.Mathf.Clamp01(0.40f * f.BrowFurrowNorm + 0.30f * (1f - f.EarNorm) + 0.30f * f.MouthTensionNorm);
                float F = UnityEngine.Mathf.Clamp01(0.35f * f.BrowFurrowNorm + 0.25f * (1f - f.EarNorm) + 0.40f * f.MouthTensionNorm);
                w.WriteLine(string.Join(",", i, f.BrowFurrowNorm.ToString("R", Inv), f.EarNorm.ToString("R", Inv),
                    f.MouthTensionNorm.ToString("R", Inv), f.YawRatio.ToString("R", Inv), f.PitchRatio.ToString("R", Inv),
                    f.PoseGateActive ? 1 : 0, C.ToString("R", Inv), F.ToString("R", Inv)));
            }
        }
        // timing: warm up, then 200 passes over all landmark sets
        for (int i = 0; i < n; i++) AffectFeatureExtractor.Extract(sets[i]);
        var sw = Stopwatch.StartNew(); int reps = 200; float sink = 0;
        for (int r = 0; r < reps; r++) for (int i = 0; i < n; i++) sink += AffectFeatureExtractor.Extract(sets[i]).BrowFurrowNorm;
        sw.Stop();
        Console.WriteLine($"features: {n} sets; mean extract time {sw.Elapsed.TotalMilliseconds * 1000.0 / (reps * n):F3} us (sink {sink:F1})");
    }

    static void SetField(object o, string name, object v) =>
        o.GetType().GetField(name, BindingFlags.NonPublic | BindingFlags.Instance).SetValue(o, v);

    static void Controller(string dir)
    {
        var tut = new A3R_ResistorTutorialController { CurrentState = ResistorTutorialState.RESISTOR_TARGET };
        UnityEngine.Object.Registry[typeof(A3R_ResistorTutorialController)] = tut;
        var traces = new SortedDictionary<int, List<(float t, float c)>>();
        foreach (var line in File.ReadLines(Path.Combine(dir, "traces.csv")))
        {
            if (line.StartsWith("trace")) continue;
            var p = line.Split(',');
            int k = int.Parse(p[0], Inv);
            if (!traces.TryGetValue(k, out var l)) traces[k] = l = new List<(float, float)>();
            l.Add((float.Parse(p[1], Inv), float.Parse(p[2], Inv)));
        }
        long updates = 0; var sw = new Stopwatch();
        using (var w = new StreamWriter(Path.Combine(dir, "cs_fires.csv")))
        {
            w.WriteLine("trace,t_fire");
            foreach (var kv in traces)
            {
                var ctl = new AdaptationController();
                SetField(ctl, "confusionThreshold", 0.40f); SetField(ctl, "frustrationThreshold", 0.65f);
                SetField(ctl, "dwellTimeSeconds", 3.0f); SetField(ctl, "cooldownSeconds", 15.0f);
                SetField(ctl, "maxAdaptationsPerStep", 3);
                int trace = kv.Key;
                ctl.OnAdaptationTriggered += e => w.WriteLine($"{trace},{UnityEngine.Time.time.ToString("R", Inv)}");
                foreach (var (t, c) in kv.Value)
                {
                    UnityEngine.Time.time = t;
                    sw.Start(); ctl.UpdateAffectScores(c, 0f, 0f, "step"); sw.Stop();
                    updates++;
                }
            }
        }
        Console.WriteLine($"controller: {traces.Count} traces, {updates} updates; mean update time {sw.Elapsed.TotalMilliseconds * 1000.0 / updates:F3} us");
    }
}
