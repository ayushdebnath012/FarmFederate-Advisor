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

  /// Get the friendly name for any model key
  String _getFriendlyName(String key) {
    return LLM_FRIENDLY_NAMES[key] ??
        VIT_FRIENDLY_NAMES[key] ??
        FUSION_FRIENDLY_NAMES[key] ??
        key;
  }

  /// Get the friendly description for any model key
  String _getFriendlyDescription(String key) {
    return LLM_FRIENDLY_DESCRIPTIONS[key] ??
        VIT_FRIENDLY_DESCRIPTIONS[key] ??
        FUSION_FRIENDLY_DESCRIPTIONS[key] ??
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
            const SizedBox(height: 16),
            _buildSectionTitle('How We Read Your Description', Icons.text_fields),
            const SizedBox(height: 8),
            _buildFriendlyChipSelector(LLM_ENCODERS, _selectedLlm, (v) {
              setState(() => _selectedLlm = v);
            }),
            const SizedBox(height: 16),
            _buildSectionTitle('How We Check Your Photo', Icons.image),
            const SizedBox(height: 8),
            _buildFriendlyChipSelector(VIT_ENCODERS, _selectedVit, (v) {
              setState(() => _selectedVit = v);
            }),
            const SizedBox(height: 16),
            _buildSectionTitle('How We Combine Everything', Icons.merge_type),
            const SizedBox(height: 8),
            _buildFriendlyChipSelector(VLM_FUSIONS, _selectedFusion, (v) {
              setState(() => _selectedFusion = v);
            }),
            const SizedBox(height: 24),
            _buildCurrentSelection(),
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
              'Choose how your crops are analyzed. The recommended settings work best for most farmers.',
              style: TextStyle(color: Colors.white70),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String title, IconData icon) {
    return Row(
      children: [
        Icon(icon, color: AppTheme.primaryGreen, size: 20),
        const SizedBox(width: 8),
        Text(
          title,
          style: const TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.bold,
            fontSize: 16,
          ),
        ),
      ],
    );
  }

  Widget _buildFriendlyChipSelector(
      List<String> options, String selected, ValueChanged<String> onSelected) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: options.map((option) {
        final isSelected = option == selected;
        final isBest = _isBest(option);
        final friendlyName = _getFriendlyName(option);
        final label = isBest ? '$friendlyName (Best)' : friendlyName;
        return Tooltip(
          message: _getFriendlyDescription(option),
          child: ChoiceChip(
            label: Text(
              label,
              style: TextStyle(
                color: isSelected ? Colors.white : Colors.white70,
                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
              ),
            ),
            selected: isSelected,
            selectedColor: AppTheme.primaryGreen,
            backgroundColor: AppTheme.cardDark,
            onSelected: (_) => onSelected(option),
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
          _buildConfigRow('Description Reader', _getFriendlyName(_selectedLlm)),
          _buildConfigRow('Photo Scanner', _getFriendlyName(_selectedVit)),
          _buildConfigRow('Combination Method', _getFriendlyName(_selectedFusion)),
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
                    'Recommended settings - 85% accuracy',
                    style: TextStyle(color: AppTheme.primaryGreen, fontSize: 12),
                  ),
                ],
              ),
            )
          else
            const Text(
              'Tip: The "Best" options give the most accurate results',
              style: TextStyle(color: Colors.white54, fontSize: 12),
            ),
        ],
      ),
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
