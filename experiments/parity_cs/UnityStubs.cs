// Minimal stand-ins for the UnityEngine API surface used by
// AffectFeatureExtractor.cs, AdaptationController.cs and AdaptationEvents.cs.
// Arithmetic follows UnityEngine (single precision; Vector3.Distance as in Unity).
using System;
using System.Collections.Generic;

namespace UnityEngine
{
    public struct Vector3
    {
        public float x, y, z;
        public Vector3(float x, float y, float z) { this.x = x; this.y = y; this.z = z; }
        public static float Distance(Vector3 a, Vector3 b)
        {
            float dx = a.x - b.x, dy = a.y - b.y, dz = a.z - b.z;
            return (float)Math.Sqrt(dx * dx + dy * dy + dz * dz);
        }
    }

    public static class Mathf
    {
        public static float Clamp01(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);
        public static float Abs(float v) => Math.Abs(v);
        public static int Min(int a, int b) => a < b ? a : b;
        public static float Min(float a, float b) => a < b ? a : b;
        public static float Max(float a, float b) => a > b ? a : b;
    }

    public static class Time { public static float time; public static float unscaledDeltaTime; }

    public static class Debug
    {
        public static bool Quiet = true;
        public static void Log(object m) { if (!Quiet) Console.WriteLine(m); }
        public static void LogWarning(object m) { if (!Quiet) Console.WriteLine(m); }
        public static void LogError(object m) { Console.Error.WriteLine(m); }
    }

    public class Object
    {
        public static readonly Dictionary<Type, object> Registry = new Dictionary<Type, object>();
        public static T FindFirstObjectByType<T>() where T : class =>
            Registry.TryGetValue(typeof(T), out var o) ? (T)o : null;
    }

    public class Component : Object { public T GetComponent<T>() where T : class => null; }
    public class Behaviour : Component { }
    public class MonoBehaviour : Behaviour { }

    [AttributeUsage(AttributeTargets.Field)] public class SerializeField : Attribute { }
    [AttributeUsage(AttributeTargets.Field, AllowMultiple = true)] public class HeaderAttribute : Attribute { public HeaderAttribute(string s) { } }
    [AttributeUsage(AttributeTargets.Field)] public class TooltipAttribute : Attribute { public TooltipAttribute(string s) { } }
    [AttributeUsage(AttributeTargets.Field)] public class RangeAttribute : Attribute { public RangeAttribute(float a, float b) { } }
}

namespace A3R.ContentManager
{
    // Stand-ins for the two app classes referenced by AdaptationController.
    public class CanonicalEvent { public string behavior; }
    public class ARContentManager : UnityEngine.MonoBehaviour
    {
        public CanonicalEvent CurrentCanonicalEvent => null;
        public void ApplyAdaptation(A3R.Adaptation.AdaptationAction action, string behavior) { }
    }
}

namespace A3R.ARGuidance
{
    public enum ResistorTutorialState { NONE, RESISTOR_TARGET, LED_TARGET, ESP32_TARGET, WIRE1_TARGET, WIRE2_TARGET, WIRE3_TARGET, COMPLETE }
    public class A3R_ResistorTutorialController : UnityEngine.MonoBehaviour
    {
        public ResistorTutorialState CurrentState { get; set; } = ResistorTutorialState.RESISTOR_TARGET;
    }
}
