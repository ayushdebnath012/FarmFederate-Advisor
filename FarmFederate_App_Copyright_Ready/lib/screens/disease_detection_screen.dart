import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../services/api_service.dart';
import '../theme/app_theme.dart';

class DiseaseDetectionScreen extends StatefulWidget {
  final String apiBase;
  const DiseaseDetectionScreen({Key? key, required this.apiBase})
      : super(key: key);

  @override
  State<DiseaseDetectionScreen> createState() => _DiseaseDetectionScreenState();
}

class _DiseaseDetectionScreenState extends State<DiseaseDetectionScreen>
    with SingleTickerProviderStateMixin {
  late final TabController _tab;
  final _textCtrl = TextEditingController();
  final _picker = ImagePicker();
  late final DiseaseApiService _api;

  Uint8List? _annotatedImage;
  String? _displayName;
  String? _className;
  double? _confidence;
  List<String> _remedies = [];
  List<Map<String, dynamic>> _allScores = [];
  String? _profileLabel;
  String? _error;
  bool _loading = false;

  Uint8List? _pickedBytes;

  @override
  void initState() {
    super.initState();
    _tab = TabController(length: 2, vsync: this);
    _api = DiseaseApiService(widget.apiBase);
  }

  @override
  void dispose() {
    _tab.dispose();
    _textCtrl.dispose();
    super.dispose();
  }

  void _clearResult() {
    setState(() {
      _annotatedImage = null;
      _displayName = null;
      _className = null;
      _confidence = null;
      _remedies = [];
      _allScores = [];
      _profileLabel = null;
      _error = null;
    });
  }

  Future<void> _pickImage(ImageSource src) async {
    final xf = await _picker.pickImage(source: src, imageQuality: 85);
    if (xf == null) return;
    final bytes = await xf.readAsBytes();
    setState(() {
      _pickedBytes = bytes;
      _clearResult();
    });
  }

  Future<void> _detectFromImage() async {
    if (_pickedBytes == null) return;
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final res = await _api.detectDisease(imageBytes: _pickedBytes!);
      if (res['error'] != null) {
        setState(() {
          _error = res['error'];
          _loading = false;
        });
        return;
      }
      setState(() {
        _displayName = res['display_name'] as String?;
        _className = res['class_name'] as String?;
        _confidence = (res['confidence'] as num?)?.toDouble();
        _remedies = List<String>.from(res['remedies'] ?? []);
        _allScores = (res['all_scores'] as List?)
                ?.map((e) => Map<String, dynamic>.from(e as Map))
                .toList() ??
            [];
        _profileLabel = res['model'] as String?;
        _annotatedImage = base64Decode(res['annotated_image'] as String);
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Future<void> _detectFromText() async {
    final q = _textCtrl.text.trim();
    if (q.isEmpty) return;
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final res = await _api.diseaseFromText(query: q);
      if (res['error'] != null) {
        setState(() {
          _error = res['error'];
          _loading = false;
        });
        return;
      }
      setState(() {
        _displayName = res['display_name'] as String?;
        _className = res['class_name'] as String?;
        _confidence = (res['confidence'] as num?)?.toDouble();
        _remedies = List<String>.from(res['remedies'] ?? []);
        _allScores = (res['all_scores'] as List?)
                ?.map((e) => Map<String, dynamic>.from(e as Map))
                .toList() ??
            [];
        _profileLabel = res['matched_by'] == 'description'
            ? 'Predicted from symptom description'
            : 'Matched by disease name';
        _annotatedImage = base64Decode(res['reference_image'] as String);
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _loading = false;
      });
    }
  }

  Color _classColor(String? cls) {
    switch (cls) {
      case 'LEAF_BLIGHT':
        return const Color(0xFF22A822);
      case 'LEAF_HOPPERS':
        return Colors.redAccent;
      case 'LEAF_RUST':
        return const Color(0xFF1E90FF);
      case 'LOOPER_CATERPILLARS':
        return Colors.purpleAccent;
      case 'MOSQUITO_BUG':
        return const Color(0xFF00CED1);
      default:
        return AppTheme.primaryGreen;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundDark,
      appBar: AppBar(
        backgroundColor: AppTheme.cardDark,
        elevation: 0,
        title: const Text(
          'Tea Disease Detector',
          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold),
        ),
        bottom: TabBar(
          controller: _tab,
          indicatorColor: AppTheme.primaryGreen,
          labelColor: AppTheme.primaryGreen,
          unselectedLabelColor: Colors.white54,
          tabs: const [
            Tab(icon: Icon(Icons.camera_alt), text: 'Image'),
            Tab(icon: Icon(Icons.text_fields), text: 'Text'),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tab,
        children: [_buildImageTab(), _buildTextTab()],
      ),
    );
  }

  Widget _buildImageTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          _buildImagePicker(),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            icon: _loading
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      color: Colors.white,
                      strokeWidth: 2,
                    ),
                  )
                : const Icon(Icons.search),
            label: Text(_loading ? 'Analysing...' : 'Detect Disease'),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryGreen,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
            ),
            onPressed:
                (_loading || _pickedBytes == null) ? null : _detectFromImage,
          ),
          const SizedBox(height: 20),
          if (_error != null) _buildError(),
          if (_annotatedImage != null && _displayName != null) ...[
            _buildResultImage(),
            const SizedBox(height: 16),
            _buildDiseaseCard(),
            if (_allScores.isNotEmpty) ...[
              const SizedBox(height: 12),
              _buildAllScoresCard(),
            ],
            const SizedBox(height: 16),
            _buildRemediesCard(),
          ],
        ],
      ),
    );
  }

  Widget _buildImagePicker() {
    return GestureDetector(
      onTap: () => _showImageSourceDialog(),
      child: Container(
        height: 220,
        decoration: BoxDecoration(
          color: AppTheme.cardDark,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white12),
        ),
        child: _pickedBytes != null
            ? ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.memory(
                  _pickedBytes!,
                  fit: BoxFit.cover,
                  width: double.infinity,
                ),
              )
            : const Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.add_photo_alternate_outlined,
                    color: AppTheme.primaryGreen,
                    size: 48,
                  ),
                  SizedBox(height: 12),
                  Text(
                    'Tap to pick a leaf image',
                    style: TextStyle(color: Colors.white54, fontSize: 14),
                  ),
                  SizedBox(height: 4),
                  Text(
                    'Camera or Gallery',
                    style: TextStyle(color: Colors.white38, fontSize: 12),
                  ),
                ],
              ),
      ),
    );
  }

  void _showImageSourceDialog() {
    showModalBottomSheet(
      context: context,
      backgroundColor: AppTheme.cardDark,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (_) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: const Icon(Icons.camera_alt, color: Colors.white70),
              title: const Text(
                'Camera',
                style: TextStyle(color: Colors.white),
              ),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.camera);
              },
            ),
            ListTile(
              leading: const Icon(Icons.photo_library, color: Colors.white70),
              title: const Text(
                'Gallery',
                style: TextStyle(color: Colors.white),
              ),
              onTap: () {
                Navigator.pop(context);
                _pickImage(ImageSource.gallery);
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTextTab() {
    return SingleChildScrollView(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(
              color: AppTheme.primaryGreen.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(14),
              border: Border.all(
                color: AppTheme.primaryGreen.withValues(alpha: 0.3),
              ),
            ),
            child: const Row(
              children: [
                Icon(
                  Icons.info_outline,
                  color: AppTheme.primaryGreen,
                  size: 20,
                ),
                SizedBox(width: 10),
                Expanded(
                  child: Text(
                    'Write the symptoms in your own words. The prompts below are only examples; tap one to fill the box, then edit it or replace it before identifying.',
                    style: TextStyle(color: Colors.white70, fontSize: 13),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
          const Text(
            'Example prompts',
            style: TextStyle(
              color: Colors.white70,
              fontSize: 13,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              'orange rust pustules under the leaf',
              'brown necrotic leaf patches',
              'stippling and tip burn on young leaves',
              'ragged holes with frass',
              'corky lesions on tender shoots',
            ]
                .map(
                  (e) => ActionChip(
                    label: Text(
                      e,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                      ),
                    ),
                    backgroundColor: AppTheme.cardDark,
                    side: const BorderSide(color: Colors.white24),
                    onPressed: () {
                      _textCtrl.text = e;
                      _clearResult();
                    },
                  ),
                )
                .toList(),
          ),
          const SizedBox(height: 16),
          TextField(
            controller: _textCtrl,
            minLines: 3,
            maxLines: 5,
            keyboardType: TextInputType.multiline,
            textInputAction: TextInputAction.newline,
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              hintText:
                  'Describe what you see, for example: orange pustules under leaves, ragged holes with frass, brown patches, stippling, or corky lesions.',
              hintStyle: const TextStyle(color: Colors.white38),
              filled: true,
              fillColor: AppTheme.cardDark,
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide.none,
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: const BorderSide(color: Colors.white12),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: const BorderSide(color: AppTheme.primaryGreen),
              ),
              prefixIcon: const Icon(Icons.search, color: Colors.white38),
              suffixIcon: _textCtrl.text.isNotEmpty
                  ? IconButton(
                      icon: const Icon(Icons.clear, color: Colors.white38),
                      onPressed: () {
                        _textCtrl.clear();
                        _clearResult();
                      },
                    )
                  : null,
            ),
            onChanged: (_) => setState(() {}),
          ),
          const SizedBox(height: 12),
          ElevatedButton.icon(
            icon: _loading
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(
                      color: Colors.white,
                      strokeWidth: 2,
                    ),
                  )
                : const Icon(Icons.image_search),
            label: Text(
              _loading ? 'Loading...' : 'Identify Disease & Show Image',
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryGreen,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(vertical: 14),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
              ),
            ),
            onPressed: (_loading || _textCtrl.text.trim().isEmpty)
                ? null
                : _detectFromText,
          ),
          const SizedBox(height: 20),
          if (_error != null) _buildError(),
          if (_annotatedImage != null && _displayName != null) ...[
            _buildResultImage(),
            const SizedBox(height: 16),
            _buildDiseaseCard(),
            if (_allScores.isNotEmpty) ...[
              const SizedBox(height: 12),
              _buildAllScoresCard(),
            ],
            const SizedBox(height: 16),
            _buildRemediesCard(),
          ],
        ],
      ),
    );
  }

  Widget _buildResultImage() {
    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: _classColor(_className).withValues(alpha: 0.5),
          width: 2,
        ),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(14),
        child: Image.memory(_annotatedImage!, fit: BoxFit.contain),
      ),
    );
  }

  Widget _buildDiseaseCard() {
    final color = _classColor(_className);
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(Icons.local_florist, color: color, size: 28),
          ),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _displayName ?? '',
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                if (_confidence != null) ...[
                  const SizedBox(height: 6),
                  Row(
                    children: [
                      const Text(
                        'Confidence: ',
                        style: TextStyle(color: Colors.white54, fontSize: 13),
                      ),
                      Text(
                        '${(_confidence! * 100).toStringAsFixed(1)}%',
                        style: TextStyle(
                          color: color,
                          fontSize: 13,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: _confidence,
                      backgroundColor: Colors.white12,
                      valueColor: AlwaysStoppedAnimation<Color>(color),
                      minHeight: 5,
                    ),
                  ),
                ] else
                  const Text(
                    'Reference image from reference set',
                    style: TextStyle(color: Colors.white54, fontSize: 13),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRemediesCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.healing, color: AppTheme.accentCyan, size: 20),
              SizedBox(width: 8),
              Text(
                'Recommended Remedies',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ...List.generate(_remedies.length, (i) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 10),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Container(
                    width: 24,
                    height: 24,
                    decoration: BoxDecoration(
                      color: AppTheme.primaryGreen.withValues(alpha: 0.2),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    alignment: Alignment.center,
                    child: Text(
                      '${i + 1}',
                      style: const TextStyle(
                        color: AppTheme.primaryGreen,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),
                  Expanded(
                    child: Text(
                      _remedies[i],
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 13,
                        height: 1.4,
                      ),
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildAllScoresCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.white10),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.bar_chart, color: AppTheme.accentCyan, size: 18),
              const SizedBox(width: 8),
              const Text(
                'All Class Scores',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Spacer(),
              if (_profileLabel != null)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 3,
                  ),
                  decoration: BoxDecoration(
                    color: AppTheme.primaryGreen.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                    border: Border.all(
                      color: AppTheme.primaryGreen.withValues(alpha: 0.4),
                    ),
                  ),
                  child: Text(
                    _profileLabel!,
                    style: const TextStyle(
                      color: AppTheme.primaryGreen,
                      fontSize: 10,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 12),
          ..._allScores.map((s) {
            final conf =
                ((s['confidence'] ?? s['prob'] ?? 0.0) as num).toDouble();
            final cls = (s['class'] ?? s['label'] ?? '') as String;
            final color = _classColor(cls);
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                children: [
                  SizedBox(
                    width: 130,
                    child: Text(
                      s['display'] as String? ?? cls,
                      style: const TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: conf,
                        backgroundColor: Colors.white12,
                        valueColor: AlwaysStoppedAnimation<Color>(color),
                        minHeight: 8,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  SizedBox(
                    width: 42,
                    child: Text(
                      '${(conf * 100).toStringAsFixed(1)}%',
                      style: TextStyle(
                        color: color,
                        fontSize: 11,
                        fontWeight: FontWeight.bold,
                      ),
                      textAlign: TextAlign.right,
                    ),
                  ),
                ],
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _buildError() {
    return Container(
      padding: const EdgeInsets.all(12),
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.redAccent.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.redAccent.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Colors.redAccent, size: 20),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              _error!,
              style: const TextStyle(color: Colors.redAccent, fontSize: 13),
            ),
          ),
        ],
      ),
    );
  }
}
