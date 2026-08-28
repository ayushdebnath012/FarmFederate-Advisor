import 'package:flutter/material.dart';

class FederatedLearningScreen extends StatefulWidget {
  final String apiBase;

  const FederatedLearningScreen({Key? key, required this.apiBase})
      : super(key: key);

  @override
  State<FederatedLearningScreen> createState() =>
      _FederatedLearningScreenState();
}

class _FederatedLearningScreenState extends State<FederatedLearningScreen> {
  bool _isTraining = false;
  final int _currentRound = 3;
  final int _totalRounds = 8; // T=8 as per paper
  final int _participatingFarms = 3; // K=3 clients as per paper

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF0A0E21),
      appBar: AppBar(
        elevation: 0,
        backgroundColor: const Color(0xFF1D1E33),
        title: const Text('Farm Network - Federated Learning'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildPrivacyBanner(),
            const SizedBox(height: 24),
            _buildParadigmComparison(),
            const SizedBox(height: 24),
            _buildTrainingStatus(),
            const SizedBox(height: 24),
            _buildParticipatingFarms(),
            const SizedBox(height: 24),
            _buildPrivacyGuarantees(),
            const SizedBox(height: 24),
            _buildTrainingHistory(),
          ],
        ),
      ),
    );
  }

  Widget _buildPrivacyBanner() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF134E5E), Color(0xFF71B280)],
        ),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(12),
            ),
            child: const Icon(Icons.security, color: Colors.white, size: 32),
          ),
          const SizedBox(width: 16),
          const Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Federated LLM | ViT | VLM',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 4),
                Text(
                  'Three independent paradigms trained with FedAvg - K=3 clients, T=8 rounds, E=3 local epochs. Raw farm data never leaves your device.',
                  style: TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildParadigmComparison() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'Cross-Paradigm Results',
            style: TextStyle(
                color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 4),
          const Text(
            'All paradigms trained under identical FedAvg conditions',
            style: TextStyle(color: Colors.white54, fontSize: 11),
          ),
          const SizedBox(height: 16),
          _buildParadigmCard(
            paradigm: 'Federated LLM',
            bestModel: 'BERT-tiny',
            modality: 'Text (agronomist symptom descriptions)',
            centF1: 0.427,
            fedF1: 0.460,
            retention: 107.7,
            color: const Color(0xFF4169E1),
            icon: Icons.text_fields,
            note: 'Exceeds centralized baseline in federated setting',
          ),
          const SizedBox(height: 12),
          _buildParadigmCard(
            paradigm: 'Federated ViT',
            bestModel: 'EfficientNet',
            modality: 'Image (tea leaf photographs)',
            centF1: 0.886,
            fedF1: 0.886,
            retention: 100.0,
            color: const Color(0xFF228B22),
            icon: Icons.image,
            note: 'Perfect retention - no accuracy loss from federation',
          ),
          const SizedBox(height: 12),
          _buildParadigmCard(
            paradigm: 'Federated VLM',
            bestModel: 'CLIP fusion',
            modality: 'Text + Image combined',
            centF1: 0.873,
            fedF1: 0.861,
            retention: 98.6,
            color: const Color(0xFFB22222),
            icon: Icons.merge_type,
            note: 'Best overall federated performance',
            isBest: true,
          ),
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.white.withValues(alpha: 0.05),
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Row(
              children: [
                Icon(Icons.info_outline, color: Colors.white38, size: 14),
                SizedBox(width: 8),
                Expanded(
                  child: Text(
                    'Avg Fed/Cent retention: 102.1% across all 3 paradigms',
                    style: TextStyle(color: Colors.white54, fontSize: 11),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildParadigmCard({
    required String paradigm,
    required String bestModel,
    required String modality,
    required double centF1,
    required double fedF1,
    required double retention,
    required Color color,
    required IconData icon,
    required String note,
    bool isBest = false,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
            color: color.withValues(alpha: isBest ? 0.6 : 0.3),
            width: isBest ? 1.5 : 1),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: color, size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  paradigm,
                  style: TextStyle(
                    color: color,
                    fontWeight: FontWeight.bold,
                    fontSize: 15,
                  ),
                ),
              ),
              if (isBest)
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 7, vertical: 2),
                  decoration: BoxDecoration(
                    color: color.withValues(alpha: 0.2),
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text('BEST',
                      style: TextStyle(
                          color: color,
                          fontSize: 9,
                          fontWeight: FontWeight.bold)),
                ),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            '$bestModel | $modality',
            style: const TextStyle(color: Colors.white60, fontSize: 11),
          ),
          const SizedBox(height: 10),
          Row(
            children: [
              _buildF1Chip('Cent', centF1, Colors.white54),
              const SizedBox(width: 8),
              _buildF1Chip('Fed', fedF1, color),
              const SizedBox(width: 8),
              _buildRetentionChip(retention, color),
            ],
          ),
          const SizedBox(height: 6),
          Text(note,
              style: const TextStyle(color: Colors.white38, fontSize: 10)),
        ],
      ),
    );
  }

  Widget _buildF1Chip(String label, double value, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.15),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        '$label F1 ${value.toStringAsFixed(3)}',
        style:
            TextStyle(color: color, fontSize: 11, fontWeight: FontWeight.w600),
      ),
    );
  }

  Widget _buildRetentionChip(double retention, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: Colors.white.withValues(alpha: 0.05),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        '${retention.toStringAsFixed(1)}% retained',
        style: const TextStyle(color: Colors.white54, fontSize: 11),
      ),
    );
  }

  Widget _buildTrainingStatus() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Training Status',
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: _isTraining
                      ? Colors.green.withValues(alpha: 0.2)
                      : Colors.grey.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                      color: _isTraining ? Colors.green : Colors.grey),
                ),
                child: Text(
                  _isTraining ? 'Training' : 'Idle',
                  style: TextStyle(
                    color: _isTraining ? Colors.green : Colors.grey,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 20),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Round $_currentRound of $_totalRounds  |  $_participatingFarms clients',
                style: const TextStyle(color: Colors.white70, fontSize: 14),
              ),
              Text(
                '${((_currentRound / _totalRounds) * 100).toStringAsFixed(0)}%',
                style: const TextStyle(
                    color: Colors.white,
                    fontSize: 14,
                    fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 12),
          LinearProgressIndicator(
            value: _currentRound / _totalRounds,
            backgroundColor: Colors.white12,
            valueColor: const AlwaysStoppedAnimation(Color(0xFF1D976C)),
          ),
          const SizedBox(height: 20),
          ElevatedButton.icon(
            onPressed: () => setState(() => _isTraining = !_isTraining),
            icon: Icon(_isTraining ? Icons.stop : Icons.play_arrow),
            label: Text(_isTraining ? 'Stop Training' : 'Start Training'),
            style: ElevatedButton.styleFrom(
              backgroundColor:
                  _isTraining ? Colors.red : const Color(0xFF1D976C),
              minimumSize: const Size(double.infinity, 48),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildParticipatingFarms() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                'Participating Farms',
                style: TextStyle(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.bold),
              ),
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: Colors.blue.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(20),
                ),
                child: Text(
                  '$_participatingFarms clients (K=3)',
                  style: const TextStyle(
                      color: Colors.blue,
                      fontSize: 12,
                      fontWeight: FontWeight.bold),
                ),
              ),
            ],
          ),
          const SizedBox(height: 16),
          _buildFarmCard(
              'Punjab Farm (Client 1)', 'Online', 'Contributing', Colors.green),
          const SizedBox(height: 8),
          _buildFarmCard(
              'Kerala Farm (Client 2)', 'Online', 'Training', Colors.blue),
          const SizedBox(height: 8),
          _buildFarmCard('UP Farm (Client 3)', 'Online', 'Idle', Colors.orange),
        ],
      ),
    );
  }

  Widget _buildFarmCard(
      String name, String status, String activity, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Container(
              width: 10,
              height: 10,
              decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(name,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                Text(activity,
                    style:
                        const TextStyle(color: Colors.white60, fontSize: 11)),
              ],
            ),
          ),
          Text(status,
              style: TextStyle(
                  color: color, fontSize: 12, fontWeight: FontWeight.w500)),
        ],
      ),
    );
  }

  Widget _buildPrivacyGuarantees() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Row(
            children: [
              Icon(Icons.shield, color: Colors.green, size: 24),
              SizedBox(width: 12),
              Text('Privacy Guarantees',
                  style: TextStyle(
                      color: Colors.white,
                      fontSize: 18,
                      fontWeight: FontWeight.bold)),
            ],
          ),
          const SizedBox(height: 16),
          _buildGuaranteeItem(Icons.lock_outline, 'End-to-End Encryption',
              'All communications are encrypted'),
          const SizedBox(height: 12),
          _buildGuaranteeItem(Icons.visibility_off, 'No Raw Data Sharing',
              'Only model gradients are transmitted via FedAvg'),
          const SizedBox(height: 12),
          _buildGuaranteeItem(Icons.group, 'Non-IID Dirichlet Splits',
              'Realistic heterogeneous data distribution (alpha=1.0)'),
          const SizedBox(height: 12),
          _buildGuaranteeItem(Icons.verified_user, 'Secure Aggregation',
              'Central server cannot see individual client updates'),
        ],
      ),
    );
  }

  Widget _buildGuaranteeItem(IconData icon, String title, String description) {
    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
              color: Colors.green.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(8)),
          child: Icon(icon, color: Colors.green, size: 20),
        ),
        const SizedBox(width: 12),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(title,
                  style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w600)),
              Text(description,
                  style: const TextStyle(color: Colors.white60, fontSize: 11)),
            ],
          ),
        ),
      ],
    );
  }

  Widget _buildTrainingHistory() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFF1D1E33),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('Training History',
              style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          _buildHistoryItem('Round 8', '2 hours ago', 'VLM F1: 0.861'),
          const SizedBox(height: 8),
          _buildHistoryItem('Round 5', '1 day ago', 'VLM F1: 0.849'),
          const SizedBox(height: 8),
          _buildHistoryItem('Round 1', '2 days ago', 'VLM F1: 0.821'),
        ],
      ),
    );
  }

  Widget _buildHistoryItem(String round, String time, String metric) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        children: [
          const Icon(Icons.check_circle, color: Colors.green, size: 20),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(round,
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600)),
                Text(time,
                    style:
                        const TextStyle(color: Colors.white60, fontSize: 11)),
              ],
            ),
          ),
          Text(metric,
              style: const TextStyle(
                  color: Colors.green,
                  fontSize: 13,
                  fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }
}
