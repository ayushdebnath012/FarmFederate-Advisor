import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import '../models/models.dart';

/// Simple API service that posts to /predict
class ApiService {
  final String baseUrl;
  ApiService(this.baseUrl);

  /// text: free text (optional)
  /// sensors: optional sensors string (e.g., telemetry or device readings)
  /// imageBytes: optional JPEG/PNG bytes
  Future<Map<String, dynamic>> predict({
    required String text,
    String? sensors,
    Uint8List? imageBytes,
    String? imagePath, // local path on mobile (optional)
    String? imageName,
    String clientId = "web_client",
  }) async {
    final uri = Uri.parse("$baseUrl/predict");
    // Use JSON body (server supports application/json)
    final body = {
      "text": text,
      "sensors": sensors ?? "",
      "client_id": clientId,
    };

    // If image provided (bytes or path), send multipart; otherwise JSON
    if (imageBytes != null || (imagePath != null && imagePath.isNotEmpty)) {
      final request = http.MultipartRequest('POST', uri);
      request.fields.addAll({"text": text, "sensors": sensors ?? "", "client_id": clientId});
      if (imageBytes != null) {
        request.files.add(http.MultipartFile.fromBytes('image', imageBytes, filename: imageName ?? 'upload.jpg'));
      } else {
        // fromPath may throw if file not present on web; callers should only pass imagePath on non-web platforms
        final multipart = await http.MultipartFile.fromPath('image', imagePath!);
        request.files.add(multipart);
      }
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      return json.decode(resp.body) as Map<String, dynamic>;
    } else {
      final resp = await http.post(uri,
          headers: {"Content-Type": "application/json"},
          body: json.encode(body));
      return json.decode(resp.body) as Map<String, dynamic>;
    }
  }

  /// RAG diagnose: image + description -> /rag
  Future<Map<String, dynamic>> ragDiagnose({
    required String description,
    Uint8List? imageBytes,
    String? imagePath, // local path on mobile (optional)
    String? imageName,
    String clientId = "web_client",
  }) async {
    final uri = Uri.parse("$baseUrl/rag");
    // If image provided (bytes or path), send multipart; otherwise JSON
    if (imageBytes != null || (imagePath != null && imagePath.isNotEmpty)) {
      final request = http.MultipartRequest('POST', uri);
      request.fields.addAll({"description": description, "client_id": clientId});
      if (imageBytes != null) {
        request.files.add(http.MultipartFile.fromBytes('image', imageBytes, filename: imageName ?? 'upload.jpg'));
      } else {
        final multipart = await http.MultipartFile.fromPath('image', imagePath!);
        request.files.add(multipart);
      }
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      return json.decode(resp.body) as Map<String, dynamic>;
    } else {
      final resp = await http.post(uri,
          headers: {"Content-Type": "application/json"},
          body: json.encode({"description": description, "client_id": clientId}));
      return json.decode(resp.body) as Map<String, dynamic>;
    }
  }

  /// Typed predict: returns a PredictionResult object
  Future<PredictionResult> predictTyped({
    required String text,
    String? sensors,
    Uint8List? imageBytes,
    String? imageName,
    bool estimateUncertainty = false,
    String clientId = "mobile_client",
  }) async {
    final raw = await predict(
      text: text,
      sensors: sensors,
      imageBytes: imageBytes,
      imageName: imageName,
      clientId: clientId,
    );
    return PredictionResult.fromJson(raw);
  }

  /// Get current active model info from /models endpoint
  Future<ModelInfo> getCurrentModel() async {
    try {
      final uri = Uri.parse("$baseUrl/models");
      final resp = await http.get(uri).timeout(const Duration(seconds: 10));
      final data = json.decode(resp.body);
      if (data is Map<String, dynamic> && data['current'] != null) {
        return ModelInfo.fromJson(data['current'] as Map<String, dynamic>);
      }
      if (data is Map<String, dynamic>) {
        return ModelInfo.fromJson(data);
      }
    } catch (_) {}
    // Fallback default model info
    return ModelInfo(
      id: 'default',
      name: 'Smart Crop Analysis',
      description: 'Checks your crops using photos and descriptions',
      accuracy: 0.847,
    );
  }

  /// Demo: populate Qdrant with demo points
  Future<Map<String, dynamic>> demoPopulate({int n = 3, String? collection}) async {
    final collParam = collection != null ? '&collection_name=$collection' : '';
    final uri = Uri.parse("$baseUrl/demo_populate?n=$n$collParam");
    final resp = await http.post(uri);
    return json.decode(resp.body) as Map<String, dynamic>;
  }

  /// Demo: search Qdrant using the last demo vector
  Future<Map<String, dynamic>> demoSearch({int topK = 3, String vectorType = 'visual', String? collection}) async {
    final collParam = collection != null ? '&collection_name=$collection' : '';
    final uri = Uri.parse("$baseUrl/demo_search?top_k=$topK&vector_type=$vectorType$collParam");
    final resp = await http.post(uri);
    return json.decode(resp.body) as Map<String, dynamic>;
  }
}

