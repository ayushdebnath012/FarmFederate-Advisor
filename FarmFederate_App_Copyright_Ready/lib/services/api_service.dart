import 'dart:convert';
import 'dart:typed_data';
import 'package:http/http.dart' as http;
import '../models/models.dart';

class ApiService {
  final String baseUrl;
  ApiService(this.baseUrl);

  Future<Map<String, dynamic>> predict({
    required String text,
    String? sensors,
    Uint8List? imageBytes,
    String? imagePath,
    String? imageName,
    String clientId = "web_client",
  }) async {
    final uri = Uri.parse("$baseUrl/predict");
    final body = {
      "text": text,
      "sensors": sensors ?? "",
      "client_id": clientId,
    };

    if (imageBytes != null || (imagePath != null && imagePath.isNotEmpty)) {
      final request = http.MultipartRequest('POST', uri);
      request.fields.addAll({
        "text": text,
        "sensors": sensors ?? "",
        "client_id": clientId,
      });
      if (imageBytes != null) {
        request.files.add(
          http.MultipartFile.fromBytes(
            'image',
            imageBytes,
            filename: imageName ?? 'upload.jpg',
          ),
        );
      } else {
        final multipart = await http.MultipartFile.fromPath(
          'image',
          imagePath!,
        );
        request.files.add(multipart);
      }
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      return json.decode(resp.body) as Map<String, dynamic>;
    } else {
      final resp = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: json.encode(body),
      );
      return json.decode(resp.body) as Map<String, dynamic>;
    }
  }

  Future<Map<String, dynamic>> lookupDiagnose({
    required String description,
    Uint8List? imageBytes,
    String? imagePath,
    String? imageName,
    String clientId = "web_client",
  }) async {
    final uri = Uri.parse("$baseUrl/lookup");
    if (imageBytes != null || (imagePath != null && imagePath.isNotEmpty)) {
      final request = http.MultipartRequest('POST', uri);
      request.fields.addAll({
        "description": description,
        "client_id": clientId,
      });
      if (imageBytes != null) {
        request.files.add(
          http.MultipartFile.fromBytes(
            'image',
            imageBytes,
            filename: imageName ?? 'upload.jpg',
          ),
        );
      } else {
        final multipart = await http.MultipartFile.fromPath(
          'image',
          imagePath!,
        );
        request.files.add(multipart);
      }
      final streamed = await request.send();
      final resp = await http.Response.fromStream(streamed);
      return json.decode(resp.body) as Map<String, dynamic>;
    } else {
      final resp = await http.post(
        uri,
        headers: {"Content-Type": "application/json"},
        body: json.encode({"description": description, "client_id": clientId}),
      );
      return json.decode(resp.body) as Map<String, dynamic>;
    }
  }

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

  Future<ProfileInfo> getCurrentProfile() async {
    try {
      final uri = Uri.parse("$baseUrl/models");
      final resp = await http.get(uri).timeout(const Duration(seconds: 10));
      final data = json.decode(resp.body);
      if (data is Map<String, dynamic> && data['current'] != null) {
        return ProfileInfo.fromJson(data['current'] as Map<String, dynamic>);
      }
      if (data is Map<String, dynamic>) {
        return ProfileInfo.fromJson(data);
      }
    } catch (_) {}
    return ProfileInfo(
      id: 'default',
      name: 'FarmFederate Tea Fusion',
      description:
          'fusion-based tea disease diagnostic engine using image, text, and local advisory context',
      accuracy: 0.949,
    );
  }

  Future<Map<String, dynamic>> getLookupMetrics() async {
    try {
      final uri = Uri.parse("$baseUrl/lookup/metrics");
      final resp = await http.get(uri).timeout(const Duration(seconds: 10));
      return json.decode(resp.body) as Map<String, dynamic>;
    } catch (_) {
      return {
        "sorting": {"macro_f1": 1.0, "micro_f1": 1.0},
        "retrieval": {
          "recall_at_5": 0.129,
          "mrr": 0.100,
          "ndcg_at_5": 0.172,
          "kb_coverage": 1.0,
        },
        "robustness": {"index_drift": 0.156},
        "demo_mode": true,
      };
    }
  }

  Future<List<Map<String, dynamic>>> fetchProfiles() async {
    try {
      final uri = Uri.parse("$baseUrl/models");
      final resp = await http.get(uri).timeout(const Duration(seconds: 10));
      final data = json.decode(resp.body) as Map<String, dynamic>;
      final profiles = data['models'] as List?;
      if (profiles != null) {
        return profiles.whereType<Map<String, dynamic>>().toList();
      }
    } catch (_) {}
    return [];
  }

  Future<Map<String, dynamic>> healthCheck() async {
    try {
      final uri = Uri.parse("$baseUrl/health");
      final resp = await http.get(uri).timeout(const Duration(seconds: 8));
      return json.decode(resp.body) as Map<String, dynamic>;
    } catch (_) {
      return {"status": "unreachable"};
    }
  }

  Future<Map<String, dynamic>> demoPopulate({
    int n = 3,
    String? collection,
  }) async {
    final collParam = collection != null ? '&collection_name=$collection' : '';
    final uri = Uri.parse("$baseUrl/demo_populate?n=$n$collParam");
    final resp = await http.post(uri);
    return json.decode(resp.body) as Map<String, dynamic>;
  }

  Future<Map<String, dynamic>> demoSearch({
    int topK = 3,
    String vectorType = 'visual',
    String? collection,
  }) async {
    final collParam = collection != null ? '&collection_name=$collection' : '';
    final uri = Uri.parse(
      "$baseUrl/demo_search?top_k=$topK&vector_type=$vectorType$collParam",
    );
    final resp = await http.post(uri);
    return json.decode(resp.body) as Map<String, dynamic>;
  }
}

class DiseaseApiService {
  final String baseUrl;
  DiseaseApiService(this.baseUrl);

  Future<Map<String, dynamic>> detectDisease({
    required Uint8List imageBytes,
    String imageName = 'leaf.jpg',
  }) async {
    try {
      final uri = Uri.parse("$baseUrl/disease/detect");
      final request = http.MultipartRequest('POST', uri)
        ..files.add(
          http.MultipartFile.fromBytes(
            'image',
            imageBytes,
            filename: imageName,
          ),
        );
      final streamed = await request.send().timeout(
            const Duration(seconds: 30),
          );
      final resp = await http.Response.fromStream(streamed);
      return json.decode(resp.body) as Map<String, dynamic>;
    } catch (e) {
      return {'error': 'Connection failed: $e'};
    }
  }

  Future<Map<String, dynamic>> diseaseFromText({required String query}) async {
    try {
      final uri = Uri.parse("$baseUrl/disease/text");
      final resp = await http
          .post(
            uri,
            headers: {'Content-Type': 'application/json'},
            body: json.encode({'query': query}),
          )
          .timeout(const Duration(seconds: 15));
      return json.decode(resp.body) as Map<String, dynamic>;
    } catch (e) {
      return {'error': 'Connection failed: $e'};
    }
  }
}
