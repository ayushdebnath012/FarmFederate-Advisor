import 'dart:typed_data';
import 'dart:io' show File;
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:file_picker/file_picker.dart';
import '../models/models.dart';
import '../services/api_service.dart';
import '../theme/app_theme.dart';

class MultimodalDiagnosisScreen extends StatefulWidget {
  final String apiBase;

  const MultimodalDiagnosisScreen({super.key, required this.apiBase});

  @override
  State<MultimodalDiagnosisScreen> createState() =>
      _MultimodalDiagnosisScreenState();
}

class _MultimodalDiagnosisScreenState extends State<MultimodalDiagnosisScreen> {
  final TextEditingController _symptomController = TextEditingController();
  final TextEditingController _sensorController = TextEditingController();
  final ImagePicker _picker = ImagePicker();
  late ApiService _api;

  File? _imageFile;
  Uint8List? _imageBytes;
  String? _imageName;

  bool _isLoading = false;
  PredictionResult? _result;
  ProfileInfo? _currentProfile;
  String? _error;

  bool _estimateUncertainty = false;
  bool _showAdvancedOptions = false;

  final List<Map<String, String>> _exampleSymptoms = [
    {
      'class': 'Leaf Blight',
      'text':
          'Brown water-soaked patches are spreading from the leaf edge. Some areas look necrotic with a yellow halo.',
    },
    {
      'class': 'Leaf Hoppers',
      'text':
          'Young leaves show stippling and tip burn. There are small puncture marks and marginal scorch on the flush.',
    },
    {
      'class': 'Leaf Rust',
      'text':
          'Orange yellow rust pustules are visible under the leaf. The upper surface has yellow spotting.',
    },
    {
      'class': 'Looper Caterpillars',
      'text':
          'Leaves have ragged holes and skeletonised patches. Frass is visible near the feeding damage.',
    },
    {
      'class': 'Mosquito Bug',
      'text':
          'Tender shoots have raised corky lesions with dark centres. Shoot tips are wilting and dying back.',
    },
  ];

  @override
  void initState() {
    super.initState();
    _api = ApiService(widget.apiBase);
    _loadCurrentProfile();
  }

  @override
  void dispose() {
    _symptomController.dispose();
    _sensorController.dispose();
    super.dispose();
  }

  Future<void> _loadCurrentProfile() async {
    try {
      _currentProfile = await _api.getCurrentProfile();
      if (mounted) setState(() {});
    } catch (e) {
      debugPrint('Failed to load current profile: $e');
    }
  }

  Future<void> _pickCamera() async {
    try {
      if (kIsWeb) {
        _showSnackBar(
          'Camera not available on web. Use gallery instead.',
          isError: true,
        );
        return;
      }

      final XFile? photo = await _picker.pickImage(
        source: ImageSource.camera,
        imageQuality: 85,
        maxWidth: 1024,
        maxHeight: 1024,
      );

      if (photo == null) return;

      final bytes = await photo.readAsBytes();
      setState(() {
        _imageFile = File(photo.path);
        _imageBytes = bytes;
        _imageName = photo.name;
      });
    } catch (e) {
      _showSnackBar('Camera error: $e', isError: true);
    }
  }

  Future<void> _pickGallery() async {
    try {
      final result = await FilePicker.platform.pickFiles(
        type: FileType.image,
        withData: true,
      );

      if (result == null || result.files.isEmpty) return;

      final picked = result.files.first;

      setState(() {
        _imageBytes = picked.bytes;
        _imageName = picked.name;
        if (!kIsWeb && picked.path != null) {
          _imageFile = File(picked.path!);
        } else {
          _imageFile = null;
        }
      });
    } catch (e) {
      _showSnackBar('Gallery error: $e', isError: true);
    }
  }

  void _clearImage() {
    setState(() {
      _imageFile = null;
      _imageBytes = null;
      _imageName = null;
    });
  }

  Future<void> _predict() async {
    final symptomText = _symptomController.text.trim();
    final sensorData = _sensorController.text.trim();

    if (symptomText.isEmpty && _imageBytes == null) {
      _showSnackBar(
        'Please provide a symptom description or crop image',
        isError: true,
      );
      return;
    }

    setState(() {
      _isLoading = true;
      _result = null;
      _error = null;
    });

    try {
      final result = await _api.predictTyped(
        text: symptomText,
        sensors: sensorData.isNotEmpty ? sensorData : null,
        imageBytes: _imageBytes,
        imageName: _imageName,
        estimateUncertainty: _estimateUncertainty,
      );

      setState(() {
        _result = result;
        _isLoading = false;
      });

      _showSnackBar('Diagnosis complete!', isError: false);
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isLoading = false;
      });
      _showSnackBar('Prediction failed: $e', isError: true);
    }
  }

  void _showSnackBar(String message, {required bool isError}) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: isError ? Colors.red : AppTheme.primaryGreen,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundDark,
      appBar: AppBar(
        title: const Text('Check My Crops'),
        actions: [
          IconButton(
            icon: const Icon(Icons.settings),
            onPressed: () => Navigator.pushNamed(context, '/profiles'),
            tooltip: 'Change Analysis Type',
          ),
          IconButton(
            icon: const Icon(Icons.help_outline),
            onPressed: _showInfoDialog,
            tooltip: 'Help',
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildCurrentProfileCard(),
            const SizedBox(height: 16),
            _buildImageInputSection(),
            const SizedBox(height: 16),
            _buildTextInputSection(),
            const SizedBox(height: 16),
            _buildAdvancedOptions(),
            const SizedBox(height: 16),
            _buildPredictButton(),
            const SizedBox(height: 24),
            if (_isLoading) _buildLoadingIndicator(),
            if (_result != null) _buildResultsSection(),
            if (_error != null) _buildErrorSection(),
          ],
        ),
      ),
    );
  }

  Widget _buildCurrentProfileCard() {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.primaryGreen.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: AppTheme.primaryGreen.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(
              Icons.check_circle,
              color: AppTheme.primaryGreen,
              size: 20,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Analysis Type',
                  style: TextStyle(color: Colors.white70, fontSize: 11),
                ),
                Text(
                  _currentProfile?.name ?? 'FarmFederate Tea Fusion',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            decoration: BoxDecoration(
              color: AppTheme.primaryGreen.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Text(
              '${((_currentProfile?.accuracy ?? 0.949) * 100).toStringAsFixed(0)}% F1',
              style: const TextStyle(
                color: AppTheme.primaryGreen,
                fontSize: 12,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImageInputSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.camera_alt, color: AppTheme.accentCyan, size: 20),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Take a Photo of Your Tea Leaf',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (_imageBytes != null) ...[
            Stack(
              children: [
                Container(
                  height: 200,
                  width: double.infinity,
                  decoration: BoxDecoration(
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(color: AppTheme.primaryGreen),
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(12),
                    child: Image.memory(_imageBytes!, fit: BoxFit.cover),
                  ),
                ),
                Positioned(
                  top: 8,
                  right: 8,
                  child: IconButton(
                    icon: const Icon(Icons.close),
                    style: IconButton.styleFrom(
                      backgroundColor: Colors.black54,
                      foregroundColor: Colors.white,
                    ),
                    onPressed: _clearImage,
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
          ],
          Row(
            children: [
              Expanded(
                child: OutlinedButton.icon(
                  icon: const Icon(Icons.camera_alt),
                  label: const Text('Camera'),
                  onPressed: _pickCamera,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: ElevatedButton.icon(
                  icon: const Icon(Icons.photo_library),
                  label: const Text('Gallery'),
                  onPressed: _pickGallery,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildTextInputSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.edit_note, color: AppTheme.accentBlue, size: 20),
              SizedBox(width: 8),
              Expanded(
                child: Text(
                  'Describe What You See',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 16,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _symptomController,
            maxLines: 4,
            style: const TextStyle(
              color: Colors.white,
              fontSize: 15,
              height: 1.4,
            ),
            decoration: InputDecoration(
              hintText:
                  'Tell us what you notice...\n(rust pustules, blight patches, feeding holes, corky lesions, etc.)',
              hintStyle: TextStyle(color: Colors.white.withValues(alpha: 0.4)),
              prefixIcon: const Icon(Icons.description),
              filled: true,
              fillColor: const Color(0xFF1A1A2E),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: Colors.white24),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: Colors.white24),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(12),
                borderSide: const BorderSide(color: AppTheme.primaryGreen),
              ),
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Tap an example to use it:',
            style: TextStyle(color: Colors.white70, fontSize: 12),
          ),
          const SizedBox(height: 8),
          SizedBox(
            height: 36,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              itemCount: _exampleSymptoms.length,
              itemBuilder: (context, index) {
                final example = _exampleSymptoms[index];
                return Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: ActionChip(
                    label: Text(example['class']!),
                    onPressed: () {
                      _symptomController.text = example['text']!;
                    },
                    backgroundColor: _getStressColor(
                      example['class']!,
                    ).withValues(alpha: 0.2),
                    labelStyle: TextStyle(
                      color: _getStressColor(example['class']!),
                      fontSize: 12,
                    ),
                  ),
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  Color _getStressColor(String stressClass) {
    switch (stressClass.toLowerCase()) {
      case 'leaf blight':
      case 'leaf_blight':
        return const Color(0xFF22A822);
      case 'leaf hoppers':
      case 'leaf_hoppers':
        return Colors.redAccent;
      case 'leaf rust':
      case 'leaf_rust':
        return const Color(0xFF1E90FF);
      case 'looper caterpillars':
      case 'looper_caterpillars':
        return Colors.purpleAccent;
      case 'mosquito bug':
      case 'mosquito_bug':
        return const Color(0xFF00CED1);
      default:
        return Colors.grey;
    }
  }

  Widget _buildAdvancedOptions() {
    return Container(
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          ListTile(
            leading: const Icon(Icons.tune, color: Colors.white70),
            title: const Text(
              'More Options',
              style: TextStyle(color: Colors.white),
            ),
            trailing: Icon(
              _showAdvancedOptions ? Icons.expand_less : Icons.expand_more,
              color: Colors.white70,
            ),
            onTap: () =>
                setState(() => _showAdvancedOptions = !_showAdvancedOptions),
          ),
          if (_showAdvancedOptions) ...[
            const Divider(color: Colors.white12),
            Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  TextField(
                    controller: _sensorController,
                    style: const TextStyle(color: Colors.white, fontSize: 15),
                    decoration: InputDecoration(
                      labelText: 'Weather & Soil Info (optional)',
                      hintText: 'e.g., hot weather, dry soil, rainy week',
                      hintStyle: TextStyle(
                        color: Colors.white.withValues(alpha: 0.4),
                      ),
                      prefixIcon: const Icon(Icons.wb_sunny),
                      filled: true,
                      fillColor: const Color(0xFF1A1A2E),
                      border: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: const BorderSide(color: Colors.white24),
                      ),
                      enabledBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: const BorderSide(color: Colors.white24),
                      ),
                      focusedBorder: OutlineInputBorder(
                        borderRadius: BorderRadius.circular(12),
                        borderSide: const BorderSide(
                          color: AppTheme.primaryGreen,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(height: 16),
                  SwitchListTile(
                    title: const Text(
                      'Double Check Results',
                      style: TextStyle(color: Colors.white),
                    ),
                    subtitle: const Text(
                      'Takes longer but gives more reliable results',
                      style: TextStyle(color: Colors.white70, fontSize: 12),
                    ),
                    value: _estimateUncertainty,
                    onChanged: (v) => setState(() => _estimateUncertainty = v),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildPredictButton() {
    return Container(
      decoration: BoxDecoration(
        gradient: AppTheme.primaryGradient,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: AppTheme.primaryGreen.withValues(alpha: 0.3),
            blurRadius: 12,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ElevatedButton.icon(
        icon: const Icon(Icons.search, size: 24),
        label: const Text(
          'Check My Crop',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
        style: ElevatedButton.styleFrom(
          backgroundColor: Colors.transparent,
          shadowColor: Colors.transparent,
          padding: const EdgeInsets.symmetric(vertical: 16),
        ),
        onPressed: _isLoading ? null : _predict,
      ),
    );
  }

  Widget _buildLoadingIndicator() {
    return Container(
      padding: const EdgeInsets.all(24),
      child: Column(
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 16),
          Text(
            'Analyzing your crop...',
            style: TextStyle(color: Colors.white.withValues(alpha: 0.7)),
          ),
          const SizedBox(height: 8),
          Text(
            'This will take just a moment',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.5),
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildResultsSection() {
    if (_result == null) return const SizedBox.shrink();

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        _buildResultHeader(),
        const SizedBox(height: 16),
        _buildPredictionScores(),
        const SizedBox(height: 16),
        _buildAdviceCard(),
        if (_result!.uncertainty != null) ...[
          const SizedBox(height: 16),
          _buildUncertaintyCard(),
        ],
      ],
    );
  }

  Widget _buildResultHeader() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _result!.severityColor.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _result!.severityColor),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: _result!.severityColor.withValues(alpha: 0.3),
              shape: BoxShape.circle,
            ),
            child: Icon(
              Icons.warning_amber,
              color: _result!.severityColor,
              size: 32,
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Tea Disease Identified',
                  style: TextStyle(
                    color: _result!.severityColor,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Text(
                  'See recommendations below',
                  style: TextStyle(color: Colors.white70),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPredictionScores() {
    final sortedScores = _result!.topN(5);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'What We Found',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'The higher the percentage, the more likely',
            style: TextStyle(
              color: Colors.white.withValues(alpha: 0.6),
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 16),
          ...sortedScores.asMap().entries.map((entry) {
            final index = entry.key;
            final score = entry.value;
            return _buildScoreBar(score, isTop: index == 0);
          }),
        ],
      ),
    );
  }

  Widget _buildScoreBar(ClassScore score, {bool isTop = false}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                score.stressClass.icon,
                color: score.stressClass.color,
                size: 20,
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  score.stressClass.displayName,
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: isTop ? FontWeight.bold : FontWeight.normal,
                  ),
                ),
              ),
              Text(
                '${score.probabilityPercent.toStringAsFixed(1)}%',
                style: TextStyle(
                  color: score.stressClass.color,
                  fontWeight: FontWeight.bold,
                ),
              ),
              if (score.isActive) ...[
                const SizedBox(width: 8),
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 6,
                    vertical: 2,
                  ),
                  decoration: BoxDecoration(
                    color: Colors.red.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: const Text(
                    'ACTIVE',
                    style: TextStyle(
                      color: Colors.red,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ],
            ],
          ),
          const SizedBox(height: 4),
          Stack(
            children: [
              Container(
                height: 8,
                decoration: BoxDecoration(
                  color: Colors.white12,
                  borderRadius: BorderRadius.circular(4),
                ),
              ),
              FractionallySizedBox(
                widthFactor: score.probability,
                child: Container(
                  height: 8,
                  decoration: BoxDecoration(
                    color: score.stressClass.color,
                    borderRadius: BorderRadius.circular(4),
                    boxShadow: isTop
                        ? [
                            BoxShadow(
                              color: score.stressClass.color.withValues(
                                alpha: 0.5,
                              ),
                              blurRadius: 8,
                            ),
                          ]
                        : null,
                  ),
                ),
              ),
              Positioned(
                left: score.threshold * MediaQuery.of(context).size.width * 0.7,
                child: Container(width: 2, height: 8, color: Colors.white54),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildAdviceCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.accentBlue.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.accentBlue.withValues(alpha: 0.3)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.lightbulb, color: AppTheme.accentBlue),
              SizedBox(width: 8),
              Text(
                'Recommended Actions',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            _result!.advice.isNotEmpty
                ? _result!.advice
                : 'Continue routine monitoring and maintain current practices.',
            style: const TextStyle(color: Colors.white, height: 1.5),
          ),
        ],
      ),
    );
  }

  Widget _buildUncertaintyCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.analytics_outlined, color: Colors.white70),
              SizedBox(width: 8),
              Text(
                'Uncertainty Estimation',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          ..._result!.uncertainty!.entries.map((entry) {
            final stressClass = StressClassExtension.fromLabel(entry.key);
            return Padding(
              padding: const EdgeInsets.only(bottom: 8),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Icon(
                        stressClass.icon,
                        size: 16,
                        color: stressClass.color,
                      ),
                      const SizedBox(width: 8),
                      Text(
                        stressClass.displayName,
                        style: const TextStyle(color: Colors.white70),
                      ),
                    ],
                  ),
                  Text(
                    '+/-${(entry.value * 100).toStringAsFixed(2)}%',
                    style: const TextStyle(
                      color: Colors.white,
                      fontFamily: 'monospace',
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

  Widget _buildErrorSection() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.red.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.red.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          const Icon(Icons.error_outline, color: Colors.red),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _error ?? 'An unknown error occurred',
              style: const TextStyle(color: Colors.red),
            ),
          ),
          TextButton(onPressed: _predict, child: const Text('Retry')),
        ],
      ),
    );
  }

  void _showInfoDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        backgroundColor: AppTheme.cardDark,
        title: const Text('How to Use'),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'Check your crops in 3 easy steps:',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              const SizedBox(height: 16),
              _buildHelpStep(
                '1',
                'Take a Photo',
                'Snap a clear picture of the affected tea leaf or shoot',
              ),
              const SizedBox(height: 12),
              _buildHelpStep(
                '2',
                'Describe It',
                'Tell us what you see - rust, blight, holes, stippling, or corky wounds',
              ),
              const SizedBox(height: 12),
              _buildHelpStep(
                '3',
                'Get Results',
                'Tap "Check My Crop" for disease probabilities and remedies',
              ),
              const SizedBox(height: 20),
              const Text(
                'We can detect:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              const Text('- Leaf blight'),
              const Text('- Leaf hoppers'),
              const Text('- Leaf rust'),
              const Text('- Looper caterpillars'),
              const Text('- Mosquito bug'),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: AppTheme.primaryGreen.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Row(
                  children: [
                    Icon(Icons.lock, color: AppTheme.primaryGreen, size: 20),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Your photos stay private on your device',
                        style: TextStyle(
                          color: AppTheme.primaryGreen,
                          fontSize: 13,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Got It'),
          ),
        ],
      ),
    );
  }

  Widget _buildHelpStep(String number, String title, String description) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 28,
          height: 28,
          decoration: BoxDecoration(
            color: AppTheme.primaryGreen,
            borderRadius: BorderRadius.circular(14),
          ),
          child: Center(
            child: Text(
              number,
              style: const TextStyle(
                color: Colors.white,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                ),
              ),
              Text(
                description,
                style: TextStyle(
                  color: Colors.white.withValues(alpha: 0.7),
                  fontSize: 13,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
