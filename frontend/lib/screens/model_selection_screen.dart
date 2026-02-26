import 'package:flutter/material.dart';
import '../theme/app_theme.dart';
import '../constants.dart';

/// Analysis Settings Screen
/// Lets farmers choose how their crops are analyzed using simple descriptions
class ModelSelectionScreen extends StatefulWidget {
  final String apiBase;

  const ModelSelectionScreen({super.key, required this.apiBase});

  @override
  State<ModelSelectionScreen> createState() => _ModelSelectionScreenState();
}

class _ModelSelectionScreenState extends State<ModelSelectionScreen> {
  String _selectedLlm = BEST_LLM;
  String _selectedVit = BEST_VIT;
  String _selectedFusion = BEST_FUSION;
  String? _selectedFederated;

  /// Get the friendly name for any model key
  String _getFriendlyName(String key) {
    return LLM_FRIENDLY_NAMES[key] ??
        VIT_FRIENDLY_NAMES[key] ??
        FUSION_FRIENDLY_NAMES[key] ??
        FEDERATED_FRIENDLY_NAMES[key] ??
        key;
  }

  /// Get the friendly description for any model key
  String _getFriendlyDescription(String key) {
    return LLM_FRIENDLY_DESCRIPTIONS[key] ??
        VIT_FRIENDLY_DESCRIPTIONS[key] ??
        FUSION_FRIENDLY_DESCRIPTIONS[key] ??
        FEDERATED_FRIENDLY_DESCRIPTIONS[key] ??
        '';
  }

  /// Check if this option is the recommended one
  bool _isBest(String key) {
    return key == BEST_LLM || key == BEST_VIT || key == BEST_FUSION;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.backgroundDark,
      appBar: AppBar(
        title: const Text('Analysis Settings'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildInfoCard(),
            const SizedBox(height: 20),
            _buildSectionTitle('Federated LLM  —  Text Analysis', Icons.text_fields,
                subtitle: 'ALBERT-tiny · Fed F1 0.548 · 92.9% retention'),
            const SizedBox(height: 8),
            _buildModelCards(LLM_ENCODERS, _selectedLlm, (v) {
              setState(() => _selectedLlm = v);
            }),
            const SizedBox(height: 20),
            _buildSectionTitle('Federated ViT  —  Image Analysis', Icons.image,
                subtitle: 'ConvNeXT-tiny · Fed F1 0.659 · 86.1% retention'),
            const SizedBox(height: 8),
            _buildModelCards(VIT_ENCODERS, _selectedVit, (v) {
              setState(() => _selectedVit = v);
            }),
            const SizedBox(height: 20),
            _buildSectionTitle('Federated VLM  —  Multimodal Fusion', Icons.merge_type,
                subtitle: 'CLIP fusion · Fed F1 0.785 · 92.6% retention'),
            const SizedBox(height: 8),
            _buildModelCards(VLM_FUSIONS, _selectedFusion, (v) {
              setState(() => _selectedFusion = v);
            }),
            const SizedBox(height: 20),
            _buildSectionTitle('Active Federated Paradigm', Icons.people,
                subtitle: 'K=3 clients · T=8 rounds · FedAvg · no raw data shared'),
            const SizedBox(height: 8),
            _buildModelCards(FEDERATED_MODES, _selectedFederated ?? '', (v) {
              setState(() {
                _selectedFederated = _selectedFederated == v ? null : v;
              });
            }),
            const SizedBox(height: 24),
            _buildCurrentSelection(),
            const SizedBox(height: 16),
            _buildApplyButton(),
            const SizedBox(height: 16),
            _buildNoveltyCard(),
          ],
        ),
      ),
    );
  }

  Widget _buildInfoCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.accentBlue.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.accentBlue.withValues(alpha: 0.3)),
      ),
      child: const Row(
        children: [
          Icon(Icons.info_outline, color: AppTheme.accentBlue),
          SizedBox(width: 12),
          Expanded(
            child: Text(
              'Three federated paradigms: Federated LLM (text), Federated ViT (image), Federated VLM (text+image). BEST badges show the top performer from the paper.',
              style: TextStyle(color: Colors.white70),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title, IconData icon, {String? subtitle}) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(icon, color: AppTheme.primaryGreen, size: 20),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ),
          ],
        ),
        if (subtitle != null)
          Padding(
            padding: const EdgeInsets.only(left: 28, top: 2),
            child: Text(
              subtitle,
              style: const TextStyle(color: Colors.white38, fontSize: 11),
            ),
          ),
      ],
    );
  }

  Widget _buildModelCards(
      List<String> options, String selected, ValueChanged<String> onSelected) {
    return Column(
      children: options.map((option) {
        final isSelected = option == selected;
        final isBest = _isBest(option);
        final friendlyName = _getFriendlyName(option);
        final description = _getFriendlyDescription(option);

        return Padding(
          padding: const EdgeInsets.only(bottom: 8),
          child: InkWell(
            onTap: () => onSelected(option),
            borderRadius: BorderRadius.circular(12),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
              decoration: BoxDecoration(
                color: isSelected
                    ? AppTheme.primaryGreen.withValues(alpha: 0.15)
                    : AppTheme.cardDark,
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: isSelected
                      ? AppTheme.primaryGreen
                      : Colors.white12,
                  width: isSelected ? 1.5 : 1,
                ),
              ),
              child: Row(
                children: [
                  // Selection indicator
                  Container(
                    width: 22,
                    height: 22,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: isSelected
                          ? AppTheme.primaryGreen
                          : Colors.transparent,
                      border: Border.all(
                        color: isSelected
                            ? AppTheme.primaryGreen
                            : Colors.white38,
                        width: 2,
                      ),
                    ),
                    child: isSelected
                        ? const Icon(Icons.check, size: 14, color: Colors.white)
                        : null,
                  ),
                  const SizedBox(width: 12),
                  // Name and description
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Text(
                              friendlyName,
                              style: TextStyle(
                                color: isSelected ? Colors.white : Colors.white70,
                                fontWeight: FontWeight.bold,
                                fontSize: 14,
                              ),
                            ),
                            if (isBest) ...[
                              const SizedBox(width: 6),
                              Container(
                                padding: const EdgeInsets.symmetric(
                                    horizontal: 6, vertical: 2),
                                decoration: BoxDecoration(
                                  color: AppTheme.primaryGreen.withValues(alpha: 0.3),
                                  borderRadius: BorderRadius.circular(4),
                                ),
                                child: const Text(
                                  'BEST',
                                  style: TextStyle(
                                    color: AppTheme.primaryGreen,
                                    fontSize: 9,
                                    fontWeight: FontWeight.bold,
                                  ),
                                ),
                              ),
                            ],
                          ],
                        ),
                        const SizedBox(height: 2),
                        Text(
                          description,
                          style: TextStyle(
                            color: isSelected ? Colors.white60 : Colors.white38,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      }).toList(),
    );
  }

  Widget _buildCurrentSelection() {
    final isRecommended = _selectedLlm == BEST_LLM &&
        _selectedVit == BEST_VIT &&
        _selectedFusion == BEST_FUSION;

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: AppTheme.cardDark,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.primaryGreen.withValues(alpha: 0.3)),
      ),
      child: Column(
        children: [
          const Text(
            'Your Settings',
            style: TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
              fontSize: 16,
            ),
          ),
          const SizedBox(height: 12),
          _buildConfigRow('Federated LLM', _getFriendlyName(_selectedLlm)),
          _buildConfigRow('Federated ViT', _getFriendlyName(_selectedVit)),
          _buildConfigRow('Federated VLM', _getFriendlyName(_selectedFusion)),
          if (_selectedFederated != null)
            _buildConfigRow(
                'Active Paradigm', _getFriendlyName(_selectedFederated!)),
          const SizedBox(height: 12),
          if (isRecommended)
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: AppTheme.primaryGreen.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(8),
              ),
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.check_circle, color: AppTheme.primaryGreen, size: 16),
                  SizedBox(width: 6),
                  Text(
                    'Best paradigm config — VLM Fed F1 = 0.785',
                    style: TextStyle(color: AppTheme.primaryGreen, fontSize: 12),
                  ),
                ],
              ),
            )
          else
            const Text(
              'Tip: The "BEST" options give the most accurate results',
              style: TextStyle(color: Colors.white54, fontSize: 12),
            ),
        ],
      ),
    );
  }

  Widget _buildApplyButton() {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: () {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                'Settings applied: ${_getFriendlyName(_selectedLlm)} + '
                '${_getFriendlyName(_selectedVit)} + '
                '${_getFriendlyName(_selectedFusion)}'
                '${_selectedFederated != null ? ' + ${_getFriendlyName(_selectedFederated!)}' : ''}',
              ),
              behavior: SnackBarBehavior.floating,
              backgroundColor: AppTheme.primaryGreen,
            ),
          );
          Navigator.pop(context);
        },
        icon: const Icon(Icons.check_circle, size: 20),
        label: const Text('Apply Settings', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
        style: ElevatedButton.styleFrom(
          backgroundColor: AppTheme.primaryGreen,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
      ),
    );
  }

  Widget _buildNoveltyCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          colors: [
            AppTheme.primaryGreen.withValues(alpha: 0.15),
            AppTheme.accentBlue.withValues(alpha: 0.15),
          ],
        ),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.primaryGreen.withValues(alpha: 0.2)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.star, color: Colors.amber, size: 20),
              SizedBox(width: 8),
              Text(
                'What Makes FarmFederate Special',
                style: TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _buildFeatureItem(
            Icons.text_fields,
            'Federated LLM — Text Paradigm',
            'ALBERT-tiny achieves Fed F1 = 0.548 with 92.9% retention. Most robust to non-IID data — ideal for bandwidth-limited farms.',
          ),
          const SizedBox(height: 8),
          _buildFeatureItem(
            Icons.image,
            'Federated ViT — Image Paradigm',
            'ConvNeXT-tiny achieves Fed F1 = 0.659 with 86.1% retention. Highest unimodal accuracy; most sensitive to data heterogeneity across farms.',
          ),
          const SizedBox(height: 8),
          _buildFeatureItem(
            Icons.merge_type,
            'Federated VLM — Multimodal Paradigm',
            'CLIP fusion achieves Fed F1 = 0.785 with 92.6% retention. Best overall — contrastive alignment outperforms Q-Former architectures.',
          ),
          const SizedBox(height: 8),
          _buildFeatureItem(
            Icons.sensors,
            'Real-Time IoT Sensor Fusion',
            'Combines live sensor data (soil, weather, NPK) with AI diagnosis - no other app connects IoT sensors directly to crop health analysis',
          ),
          const SizedBox(height: 8),
          _buildFeatureItem(
            Icons.science,
            'Research-Grade + Farmer-Friendly',
            'Bridges the gap between academic research tools and simple farmer apps - powerful enough for researchers, easy enough for any farmer',
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureItem(IconData icon, String title, String description) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, color: AppTheme.accentBlue, size: 18),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w600,
                  fontSize: 13,
                ),
              ),
              Text(
                description,
                style: const TextStyle(color: Colors.white54, fontSize: 11),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildConfigRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: Colors.white70)),
          Text(
            value,
            style: const TextStyle(
              color: Colors.white,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}
