#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lily v8.0 — Account Cog

Commands: /balance, /account_info, /quests.
"""

from __future__ import annotations
import discord
from discord.ext import commands
from discord import app_commands

from pollinations import PollinationsAPI
from config import POLLINATIONS_KEY


class AccountCog(commands.Cog, name="Account"):
    """Account and billing commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="balance", description="Check your Pollinations API balance")
    async def balance(self, interaction: discord.Interaction):
        """Check the pollen balance for the configured API key."""
        if not POLLINATIONS_KEY:
            await interaction.response.send_message(
                "❌ No API key configured. Set `POLLINATIONS_KEY` in .env to use this command.",
                ephemeral=True,
            )
            return

        await interaction.response.defer(thinking=True)

        api: PollinationsAPI = self.bot.api  # type: ignore

        try:
            balance_data = await api.account_balance()
            key_info = await api.account_key_info()

            embed = discord.Embed(
                title="💰 API Balance",
                color=discord.Color.green(),
            )

            # Balance
            balance = balance_data.get("balance", 0)
            embed.add_field(name="Pollen Balance", value=f"{balance:,.2f}", inline=True)

            # Key info
            key_name = key_info.get("name", "Unknown")
            key_type = key_info.get("type", "unknown")
            key_valid = key_info.get("valid", False)
            expires = key_info.get("expiresAt", "Never")

            embed.add_field(name="Key Name", value=key_name, inline=True)
            embed.add_field(name="Key Type", value=f"`{key_type}`", inline=True)
            embed.add_field(name="Valid", value="✅" if key_valid else "❌", inline=True)
            embed.add_field(name="Expires", value=expires or "Never", inline=True)

            # Budget
            budget = key_info.get("pollenBudget")
            if budget is not None:
                embed.add_field(name="Budget", value=f"{budget:,.2f} pollen", inline=True)

            # Rate limit
            rate_limited = key_info.get("rateLimitEnabled", False)
            embed.add_field(name="Rate Limited", value="Yes" if rate_limited else "No", inline=True)

            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to check balance: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="account_info", description="Show your Pollinations account info")
    async def account_info(self, interaction: discord.Interaction):
        """Show account profile information."""
        if not POLLINATIONS_KEY:
            await interaction.response.send_message(
                "❌ No API key configured.", ephemeral=True
            )
            return

        await interaction.response.defer(thinking=True)

        api: PollinationsAPI = self.bot.api  # type: ignore

        try:
            profile = await api.account_profile()

            embed = discord.Embed(
                title="👤 Account Profile",
                color=discord.Color.blue(),
            )

            if profile.get("githubUsername"):
                embed.add_field(name="GitHub", value=profile["githubUsername"], inline=True)
            if profile.get("name"):
                embed.add_field(name="Name", value=profile["name"], inline=True)
            if profile.get("email"):
                embed.add_field(name="Email", value=profile["email"], inline=True)

            community = profile.get("communityEndpointsAllowed", False)
            embed.add_field(name="Community Models", value="✅ Allowed" if community else "❌ Not allowed", inline=True)

            if profile.get("image"):
                embed.set_thumbnail(url=profile["image"])

            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to get account info: {str(e)[:200]}", ephemeral=True
            )

    @app_commands.command(name="quests", description="View available Pollinations quests")
    async def quests(self, interaction: discord.Interaction):
        """View the Pollinations quest catalog."""
        await interaction.response.defer(thinking=True)

        api: PollinationsAPI = self.bot.api  # type: ignore

        try:
            quest_data = await api.quest_catalog()
            quests = quest_data.get("quests", [])

            if not quests:
                await interaction.followup.send("No quests available right now.")
                return

            embed = discord.Embed(
                title="🎯 Quest Catalog",
                color=discord.Color.gold(),
            )

            for q in quests[:10]:
                title = q.get("title", "Unknown")
                state = q.get("state", "unknown")
                reward = q.get("rewardAmount", 0)
                category = q.get("category", "")

                state_emoji = {"available": "🟢", "completed": "✅", "coming_soon": "🔜"}.get(state, "❓")

                embed.add_field(
                    name=f"{state_emoji} {title}",
                    value=f"Category: {category} | Reward: {reward} pollen",
                    inline=False,
                )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            await interaction.followup.send(
                f"❌ Failed to get quests: {str(e)[:200]}", ephemeral=True
            )


async def setup(bot: commands.Bot):
    await bot.add_cog(AccountCog(bot))
