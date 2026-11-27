"""add job requirements

Revision ID: a76d291bfd70
Revises: 533c1aeb3473
Create Date: 2026-08-03 14:37:10.681277

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a76d291bfd70'
down_revision: str | Sequence[str] | None = '533c1aeb3473'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add structured job requirement fields."""

    op.add_column(
        "jobs",
        sa.Column(
            "required_skills",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )

    op.add_column(
        "jobs",
        sa.Column(
            "preferred_skills",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )

    op.add_column(
        "jobs",
        sa.Column(
            "languages",
            sa.JSON(),
            nullable=False,
            server_default="[]",
        ),
    )


def downgrade() -> None:
    """Remove structured job requirement fields."""

    op.drop_column("jobs", "languages")
    op.drop_column("jobs", "preferred_skills")
    op.drop_column("jobs", "required_skills")
